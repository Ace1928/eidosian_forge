import sys
from collections import defaultdict
from typing import (
from uuid import uuid4
from adagio.specs import WorkflowSpec
from triad import (
from fugue._utils.exception import modify_traceback
from fugue.collections.partition import PartitionSpec
from fugue.collections.sql import StructuredRawSQL
from fugue.collections.yielded import Yielded
from fugue.column import ColumnExpr
from fugue.column import SelectColumns as ColumnsSelect
from fugue.column import all_cols, col, lit
from fugue.constants import (
from fugue.dataframe import DataFrame, LocalBoundedDataFrame, YieldedDataFrame
from fugue.dataframe.api import is_df
from fugue.dataframe.dataframes import DataFrames
from fugue.exceptions import FugueWorkflowCompileError, FugueWorkflowError
from fugue.execution.api import engine_context
from fugue.extensions._builtins import (
from fugue.extensions.transformer.convert import _to_output_transformer, _to_transformer
from fugue.rpc import to_rpc_handler
from fugue.rpc.base import EmptyRPCHandler
from fugue.workflow._checkpoint import StrongCheckpoint, WeakCheckpoint
from fugue.workflow._tasks import Create, FugueTask, Output, Process
from fugue.workflow._workflow_context import FugueWorkflowContext
@extensible_class
class WorkflowDataFrame(DataFrame):
    """It represents the edges in the graph constructed by :class:`~.FugueWorkflow`.
    In Fugue, we use DAG to represent workflows, and the edges are strictly
    dataframes. DAG construction and execution are different steps, this class is
    used in the construction step. Although it inherits from
    :class:`~fugue.dataframe.dataframe.DataFrame`, it's not concerete data. So a
    lot of the operations are not allowed. If you want to obtain the concrete
    Fugue :class:`~fugue.dataframe.dataframe.DataFrame`, use :meth:`~.compute()`
    to execute the workflow.

    Normally, you don't construct it by yourself, you will just use the methods of it.

    :param workflow: the parent workflow it belongs to
    :param task: the task that generates this dataframe
    """

    def __init__(self, workflow: 'FugueWorkflow', task: FugueTask):
        super().__init__('_0:int')
        self._workflow = workflow
        self._task = task

    def spec_uuid(self) -> str:
        """UUID of its task spec"""
        return self._task.__uuid__()

    @property
    def native(self) -> Any:
        raise NotImplementedError

    def native_as_df(self) -> Any:
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Name of its task spec"""
        return self._task.name

    @property
    def workflow(self) -> 'FugueWorkflow':
        """The parent workflow"""
        return self._workflow

    @property
    def result(self) -> DataFrame:
        """The concrete DataFrame obtained from :meth:`~.compute()`.
        This property will not trigger compute again, but compute should
        have been called earlier and the result is cached.
        """
        return self.workflow.get_result(self)

    @property
    def partition_spec(self) -> PartitionSpec:
        """The partition spec set on the dataframe for next steps to use

        .. admonition:: Examples

            .. code-block:: python

                dag = FugueWorkflow()
                df = dag.df([[0],[1]], "a:int")
                assert df.partition_spec.empty
                df2 = df.partition(by=["a"])
                assert df.partition_spec.empty
                assert df2.partition_spec == PartitionSpec(by=["a"])
        """
        return self.metadata.get('pre_partition', PartitionSpec())

    def compute(self, *args, **kwargs) -> DataFrame:
        """Trigger the parent workflow to
        :meth:`~fugue.workflow.workflow.FugueWorkflow.run` and to generate and cache
        the result dataframe this instance represent.

        .. admonition:: Examples

            >>> df = FugueWorkflow().df([[0]],"a:int").transform(a_transformer)
            >>> df.compute().as_pandas()  # pandas dataframe
            >>> df.compute(SparkExecutionEngine).native  # spark dataframe

        .. note::

            Consider using :meth:`fugue.workflow.workflow.FugueWorkflow.run` instead.
            Because this method actually triggers the entire workflow to run, so it may
            be confusing to use this method because extra time may be taken to compute
            unrelated dataframes.

            .. code-block:: python

                dag = FugueWorkflow()
                df1 = dag.df([[0]],"a:int").transform(a_transformer)
                df2 = dag.df([[0]],"b:int")

                dag.run(SparkExecutionEngine)
                df1.result.show()
                df2.result.show()
        """
        self.workflow.run(*args, **kwargs)
        return self.result

    def process(self: TDF, using: Any, schema: Any=None, params: Any=None, pre_partition: Any=None) -> TDF:
        """Run a processor on this dataframe. It's a simple wrapper of
        :meth:`fugue.workflow.workflow.FugueWorkflow.process`

        Please read the
        :doc:`Processor Tutorial <tutorial:tutorials/extensions/processor>`

        :param using: processor-like object, if it is a string, then it must be
          the alias of a registered processor
        :param schema: |SchemaLikeObject|, defaults to None. The processor
          will be able to access this value from
          :meth:`~fugue.extensions.context.ExtensionContext.output_schema`
        :param params: |ParamsLikeObject| to run the processor, defaults to None.
          The processor will be able to access this value from
          :meth:`~fugue.extensions.context.ExtensionContext.params`
        :param pre_partition: |PartitionLikeObject|, defaults to None.
          The processor will be able to access this value from
          :meth:`~fugue.extensions.context.ExtensionContext.partition_spec`
        :return: result dataframe
        :rtype: :class:`~.WorkflowDataFrame`
        """
        if pre_partition is None:
            pre_partition = self.partition_spec
        df = self.workflow.process(self, using=using, schema=schema, params=params, pre_partition=pre_partition)
        return self._to_self_type(df)

    def output(self, using: Any, params: Any=None, pre_partition: Any=None) -> None:
        """Run a outputter on this dataframe. It's a simple wrapper of
        :meth:`fugue.workflow.workflow.FugueWorkflow.output`

        Please read the
        :doc:`Outputter Tutorial <tutorial:tutorials/extensions/outputter>`

        :param using: outputter-like object, if it is a string, then it must be
          the alias of a registered outputter
        :param params: |ParamsLikeObject| to run the outputter, defaults to None.
          The outputter will be able to access this value from
          :meth:`~fugue.extensions.context.ExtensionContext.params`
        :param pre_partition: |PartitionLikeObject|, defaults to None.
          The outputter will be able to access this value from
          :meth:`~fugue.extensions.context.ExtensionContext.partition_spec`
        """
        if pre_partition is None:
            pre_partition = self.partition_spec
        self.workflow.output(self, using=using, params=params, pre_partition=pre_partition)

    def head(self, n: int, columns: Optional[List[str]]=None) -> LocalBoundedDataFrame:
        raise NotImplementedError

    def show(self, n: int=10, with_count: bool=False, title: Optional[str]=None, best_width: int=100) -> None:
        """Show the dataframe.
        See
        :ref:`examples <tutorial:tutorials/advanced/dag:initialize a workflow>`.

        :param n: max number of rows, defaults to 10
        :param with_count: whether to show total count, defaults to False
        :param title: title to display on top of the dataframe, defaults to None
        :param best_width: max width for the output table, defaults to 100

        .. note::

            * When you call this method, it means you want the dataframe to be
              printed when the workflow executes. So the dataframe won't show until
              you run the workflow.
            * When ``with_count`` is True, it can trigger expensive calculation for
              a distributed dataframe. So if you call this function directly, you may
              need to :meth:`~.persist` the dataframe. Or you can turn on
              :ref:`tutorial:tutorials/advanced/useful_config:auto persist`
        """
        self.workflow.show(self, n=n, with_count=with_count, title=title)

    def assert_eq(self, *dfs: Any, **params: Any) -> None:
        """Wrapper of :meth:`fugue.workflow.workflow.FugueWorkflow.assert_eq` to
        compare this dataframe with other dataframes.

        :param dfs: |DataFramesLikeObject|
        :param digits: precision on float number comparison, defaults to 8
        :param check_order: if to compare the row orders, defaults to False
        :param check_schema: if compare schemas, defaults to True
        :param check_content: if to compare the row values, defaults to True
        :param no_pandas: if true, it will compare the string representations of the
          dataframes, otherwise, it will convert both to pandas dataframe to compare,
          defaults to False

        :raises AssertionError: if not equal
        """
        self.workflow.assert_eq(self, *dfs, **params)

    def assert_not_eq(self, *dfs: Any, **params: Any) -> None:
        """Wrapper of :meth:`fugue.workflow.workflow.FugueWorkflow.assert_not_eq` to
        compare this dataframe with other dataframes.

        :param dfs: |DataFramesLikeObject|
        :param digits: precision on float number comparison, defaults to 8
        :param check_order: if to compare the row orders, defaults to False
        :param check_schema: if compare schemas, defaults to True
        :param check_content: if to compare the row values, defaults to True
        :param no_pandas: if true, it will compare the string representations of the
          dataframes, otherwise, it will convert both to pandas dataframe to compare,
          defaults to False

        :raises AssertionError: if any dataframe is equal to the first dataframe
        """
        self.workflow.assert_not_eq(self, *dfs, **params)

    def select(self: TDF, *columns: Union[str, ColumnExpr], where: Optional[ColumnExpr]=None, having: Optional[ColumnExpr]=None, distinct: bool=False) -> TDF:
        """The functional interface for SQL select statement

        :param columns: column expressions, for strings they will represent
          the column names
        :param where: ``WHERE`` condition expression, defaults to None
        :param having: ``having`` condition expression, defaults to None. It
          is used when ``cols`` contains aggregation columns, defaults to None
        :param distinct: whether to return distinct result, defaults to False
        :return: the select result as a new dataframe

        .. admonition:: New Since
            :class: hint

            **0.6.0**

        .. attention::

            This interface is experimental, it's subjected to change in new versions.

        .. seealso::

            Please find more expression examples in :mod:`fugue.column.sql` and
            :mod:`fugue.column.functions`

        .. admonition:: Examples

            .. code-block:: python

                import fugue.column.functions as f
                from fugue import FugueWorkflow

                dag = FugueWorkflow()
                df = dag.df(pandas_df)

                # select existed and new columns
                df.select("a","b",lit(1,"another")))
                df.select("a",(col("b")+lit(1)).alias("x"))

                # select distinct
                df.select("a","b",lit(1,"another")),distinct=True)

                # aggregation
                # SELECT COUNT(DISTINCT *) AS x FROM df
                df.select(f.count_distinct(all_cols()).alias("x"))

                # SELECT a, MAX(b+1) AS x FROM df GROUP BY a
                df.select("a",f.max(col("b")+lit(1)).alias("x"))

                # SELECT a, MAX(b+1) AS x FROM df
                #   WHERE b<2 AND a>1
                #   GROUP BY a
                #   HAVING MAX(b+1)>0
                df.select(
                    "a",f.max(col("b")+lit(1)).alias("x"),
                    where=(col("b")<2) & (col("a")>1),
                    having=f.max(col("b")+lit(1))>0
                )
        """

        def _to_col(s: str) -> ColumnExpr:
            return col(s) if s != '*' else all_cols()
        sc = ColumnsSelect(*[_to_col(x) if isinstance(x, str) else x for x in columns], arg_distinct=distinct)
        df = self.workflow.process(self, using=Select, params=dict(columns=sc, where=where, having=having))
        return self._to_self_type(df)

    def filter(self: TDF, condition: ColumnExpr) -> TDF:
        """Filter rows by the given condition

        :param df: the dataframe to be filtered
        :param condition: (boolean) column expression
        :return: a new filtered dataframe

        .. admonition:: New Since
            :class: hint

            **0.6.0**

        .. seealso::

            Please find more expression examples in :mod:`fugue.column.sql` and
            :mod:`fugue.column.functions`

        .. admonition:: Examples

            .. code-block:: python

                import fugue.column.functions as f
                from fugue import FugueWorkflow

                dag = FugueWorkflow()
                df = dag.df(pandas_df)

                df.filter((col("a")>1) & (col("b")=="x"))
                df.filter(f.coalesce(col("a"),col("b"))>1)
        """
        df = self.workflow.process(self, using=Filter, params=dict(condition=condition))
        return self._to_self_type(df)

    def assign(self: TDF, *args: ColumnExpr, **kwargs: Any) -> TDF:
        """Update existing columns with new values and add new columns

        :param df: the dataframe to set columns
        :param args: column expressions
        :param kwargs: column expressions to be renamed to the argument names,
          if a value is not `ColumnExpr`, it will be treated as a literal
        :return: a new dataframe with the updated values

        .. tip::

            This can be used to cast data types, alter column values or add new
            columns. But you can't use aggregation in columns.

        .. admonition:: New Since
            :class: hint

            **0.6.0**

        .. seealso::

            Please find more expression examples in :mod:`fugue.column.sql` and
            :mod:`fugue.column.functions`

        .. admonition:: Examples

            .. code-block:: python

                from fugue import FugueWorkflow

                dag = FugueWorkflow()
                df = dag.df(pandas_df)

                # add/set 1 as column x
                df.assign(lit(1,"x"))
                df.assign(x=1)

                # add/set x to be a+b
                df.assign((col("a")+col("b")).alias("x"))
                df.assign(x=col("a")+col("b"))

                # cast column a data type to double
                df.assign(col("a").cast(float))

                # cast + new columns
                df.assign(col("a").cast(float),x=1,y=col("a")+col("b"))
        """
        kv: List[ColumnExpr] = [v.alias(k) if isinstance(v, ColumnExpr) else lit(v).alias(k) for k, v in kwargs.items()]
        df = self.workflow.process(self, using=Assign, params=dict(columns=list(args) + kv))
        return self._to_self_type(df)

    def aggregate(self: TDF, *agg_cols: ColumnExpr, **kwagg_cols: ColumnExpr) -> TDF:
        """Aggregate on dataframe

        :param df: the dataframe to aggregate on
        :param agg_cols: aggregation expressions
        :param kwagg_cols: aggregation expressions to be renamed to the argument names
        :return: the aggregated result as a dataframe

        .. admonition:: New Since
            :class: hint

            **0.6.0**

        .. seealso::

            Please find more expression examples in :mod:`fugue.column.sql` and
            :mod:`fugue.column.functions`

        .. admonition:: Examples

            .. code-block:: python

                import fugue.column.functions as f

                # SELECT MAX(b) AS b FROM df
                df.aggregate(f.max(col("b")))

                # SELECT a, MAX(b) AS x FROM df GROUP BY a
                df.partition_by("a").aggregate(f.max(col("b")).alias("x"))
                df.partition_by("a").aggregate(x=f.max(col("b")))
        """
        columns: List[ColumnExpr] = list(agg_cols) + [v.alias(k) for k, v in kwagg_cols.items()]
        df = self.workflow.process(self, using=Aggregate, params=dict(columns=columns), pre_partition=self.partition_spec)
        return self._to_self_type(df)

    def transform(self: TDF, using: Any, schema: Any=None, params: Any=None, pre_partition: Any=None, ignore_errors: List[Any]=_DEFAULT_IGNORE_ERRORS, callback: Any=None) -> TDF:
        """Transform this dataframe using transformer. It's a wrapper of
        :meth:`fugue.workflow.workflow.FugueWorkflow.transform`

        Please read |TransformerTutorial|

        :param using: transformer-like object, if it is a string, then it must be
          the alias of a registered transformer/cotransformer
        :param schema: |SchemaLikeObject|, defaults to None. The transformer
          will be able to access this value from
          :meth:`~fugue.extensions.context.ExtensionContext.output_schema`
        :param params: |ParamsLikeObject| to run the processor, defaults to None.
          The transformer will be able to access this value from
          :meth:`~fugue.extensions.context.ExtensionContext.params`
        :param pre_partition: |PartitionLikeObject|, defaults to None. It's
          recommended to use the equivalent wayt, which is to call
          :meth:`~.partition` and then call :meth:`~.transform` without this parameter
        :param ignore_errors: list of exception types the transformer can ignore,
          defaults to empty list
        :param callback: |RPCHandlerLikeObject|, defaults to None
        :return: the transformed dataframe
        :rtype: :class:`~.WorkflowDataFrame`

        .. note::

            :meth:`~.transform` can be lazy and will return the transformed dataframe,
            :meth:`~.out_transform` is guaranteed to execute immediately (eager) and
            return nothing
        """
        if pre_partition is None:
            pre_partition = self.partition_spec
        df = self.workflow.transform(self, using=using, schema=schema, params=params, pre_partition=pre_partition, ignore_errors=ignore_errors, callback=callback)
        return self._to_self_type(df)

    def out_transform(self: TDF, using: Any, params: Any=None, pre_partition: Any=None, ignore_errors: List[Any]=_DEFAULT_IGNORE_ERRORS, callback: Any=None) -> None:
        """Transform this dataframe using transformer. It's a wrapper of
        :meth:`fugue.workflow.workflow.FugueWorkflow.out_transform`

        Please read |TransformerTutorial|

        :param using: transformer-like object, if it is a string, then it must be
          the alias of a registered output transformer/cotransformer
        :param params: |ParamsLikeObject| to run the processor, defaults to None.
          The transformer will be able to access this value from
          :meth:`~fugue.extensions.context.ExtensionContext.params`
        :param pre_partition: |PartitionLikeObject|, defaults to None. It's
          recommended to use the equivalent wayt, which is to call
          :meth:`~.partition` and then call :meth:`~.transform` without this parameter
        :param ignore_errors: list of exception types the transformer can ignore,
          defaults to empty list
        :param callback: |RPCHandlerLikeObject|, defaults to None

        .. note::

            :meth:`~.transform` can be lazy and will return the transformed dataframe,
            :meth:`~.out_transform` is guaranteed to execute immediately (eager) and
            return nothing
        """
        if pre_partition is None:
            pre_partition = self.partition_spec
        self.workflow.out_transform(self, using=using, params=params, pre_partition=pre_partition, ignore_errors=ignore_errors, callback=callback)

    def join(self: TDF, *dfs: Any, how: str, on: Optional[Iterable[str]]=None) -> TDF:
        """Join this dataframe with dataframes. It's a wrapper of
        :meth:`fugue.workflow.workflow.FugueWorkflow.join`. |ReadJoin|

        :param dfs: |DataFramesLikeObject|
        :param how: can accept ``semi``, ``left_semi``, ``anti``, ``left_anti``,
          ``inner``, ``left_outer``, ``right_outer``, ``full_outer``, ``cross``
        :param on: it can always be inferred, but if you provide, it will be
          validated against the inferred keys. Default to None
        :return: joined dataframe
        :rtype: :class:`~.WorkflowDataFrame`
        """
        df = self.workflow.join(self, *dfs, how=how, on=on)
        return self._to_self_type(df)

    def inner_join(self: TDF, *dfs: Any, on: Optional[Iterable[str]]=None) -> TDF:
        """INNER Join this dataframe with dataframes. It's a wrapper of
        :meth:`fugue.workflow.workflow.FugueWorkflow.join`. |ReadJoin|

        :param dfs: |DataFramesLikeObject|
        :param on: it can always be inferred, but if you provide, it will be
          validated against the inferred keys. Default to None
        :return: joined dataframe
        :rtype: :class:`~.WorkflowDataFrame`
        """
        return self.join(*dfs, how='inner', on=on)

    def semi_join(self: TDF, *dfs: Any, on: Optional[Iterable[str]]=None) -> TDF:
        """LEFT SEMI Join this dataframe with dataframes. It's a wrapper of
        :meth:`fugue.workflow.workflow.FugueWorkflow.join`. |ReadJoin|

        :param dfs: |DataFramesLikeObject|
        :param on: it can always be inferred, but if you provide, it will be
          validated against the inferred keys. Default to None
        :return: joined dataframe
        :rtype: :class:`~.WorkflowDataFrame`
        """
        return self.join(*dfs, how='semi', on=on)

    def left_semi_join(self: TDF, *dfs: Any, on: Optional[Iterable[str]]=None) -> TDF:
        """LEFT SEMI Join this dataframe with dataframes. It's a wrapper of
        :meth:`fugue.workflow.workflow.FugueWorkflow.join`. |ReadJoin|

        :param dfs: |DataFramesLikeObject|
        :param on: it can always be inferred, but if you provide, it will be
          validated against the inferred keys. Default to None
        :return: joined dataframe
        :rtype: :class:`~.WorkflowDataFrame`
        """
        return self.join(*dfs, how='left_semi', on=on)

    def anti_join(self: TDF, *dfs: Any, on: Optional[Iterable[str]]=None) -> TDF:
        """LEFT ANTI Join this dataframe with dataframes. It's a wrapper of
        :meth:`fugue.workflow.workflow.FugueWorkflow.join`. |ReadJoin|

        :param dfs: |DataFramesLikeObject|
        :param on: it can always be inferred, but if you provide, it will be
          validated against the inferred keys. Default to None
        :return: joined dataframe
        :rtype: :class:`~.WorkflowDataFrame`
        """
        return self.join(*dfs, how='anti', on=on)

    def left_anti_join(self: TDF, *dfs: Any, on: Optional[Iterable[str]]=None) -> TDF:
        """LEFT ANTI Join this dataframe with dataframes. It's a wrapper of
        :meth:`fugue.workflow.workflow.FugueWorkflow.join`. |ReadJoin|

        :param dfs: |DataFramesLikeObject|
        :param on: it can always be inferred, but if you provide, it will be
          validated against the inferred keys. Default to None
        :return: joined dataframe
        :rtype: :class:`~.WorkflowDataFrame`
        """
        return self.join(*dfs, how='left_anti', on=on)

    def left_outer_join(self: TDF, *dfs: Any, on: Optional[Iterable[str]]=None) -> TDF:
        """LEFT OUTER Join this dataframe with dataframes. It's a wrapper of
        :meth:`fugue.workflow.workflow.FugueWorkflow.join`. |ReadJoin|

        :param dfs: |DataFramesLikeObject|
        :param on: it can always be inferred, but if you provide, it will be
          validated against the inferred keys. Default to None
        :return: joined dataframe
        :rtype: :class:`~.WorkflowDataFrame`
        """
        return self.join(*dfs, how='left_outer', on=on)

    def right_outer_join(self: TDF, *dfs: Any, on: Optional[Iterable[str]]=None) -> TDF:
        """RIGHT OUTER Join this dataframe with dataframes. It's a wrapper of
        :meth:`fugue.workflow.workflow.FugueWorkflow.join`. |ReadJoin|

        :param dfs: |DataFramesLikeObject|
        :param on: it can always be inferred, but if you provide, it will be
          validated against the inferred keys. Default to None
        :return: joined dataframe
        :rtype: :class:`~.WorkflowDataFrame`
        """
        return self.join(*dfs, how='right_outer', on=on)

    def full_outer_join(self: TDF, *dfs: Any, on: Optional[Iterable[str]]=None) -> TDF:
        """CROSS Join this dataframe with dataframes. It's a wrapper of
        :meth:`fugue.workflow.workflow.FugueWorkflow.join`. |ReadJoin|

        :param dfs: |DataFramesLikeObject|
        :param on: it can always be inferred, but if you provide, it will be
          validated against the inferred keys. Default to None
        :return: joined dataframe
        :rtype: :class:`~.WorkflowDataFrame`
        """
        return self.join(*dfs, how='full_outer', on=on)

    def cross_join(self: TDF, *dfs: Any) -> TDF:
        """CROSS Join this dataframe with dataframes. It's a wrapper of
        :meth:`fugue.workflow.workflow.FugueWorkflow.join`. |ReadJoin|

        :param dfs: |DataFramesLikeObject|
        :return: joined dataframe
        :rtype: :class:`~.WorkflowDataFrame`
        """
        return self.join(*dfs, how='cross')

    def union(self: TDF, *dfs: Any, distinct: bool=True) -> TDF:
        """Union this dataframe with ``dfs``.

        :param dfs: |DataFramesLikeObject|
        :param distinct: whether to perform `distinct` after union,
          default to True
        :return: unioned dataframe

        .. note::

            Currently, all dataframes in ``dfs`` must have identical schema, otherwise
            exception will be thrown.
        """
        df = self.workflow.union(self, *dfs, distinct=distinct)
        return self._to_self_type(df)

    def subtract(self: TDF, *dfs: Any, distinct: bool=True) -> TDF:
        """Subtract ``dfs`` from this dataframe.

        :param dfs: |DataFramesLikeObject|
        :param distinct: whether to perform `distinct` after subtraction,
          default to True
        :return: subtracted dataframe

        .. note::

            Currently, all dataframes in ``dfs`` must have identical schema, otherwise
            exception will be thrown.
        """
        df = self.workflow.subtract(self, *dfs, distinct=distinct)
        return self._to_self_type(df)

    def intersect(self: TDF, *dfs: Any, distinct: bool=True) -> TDF:
        """Intersect this dataframe with ``dfs``.

        :param dfs: |DataFramesLikeObject|
        :param distinct: whether to perform `distinct` after intersection,
          default to True
        :return: intersected dataframe

        .. note::

            Currently, all dataframes in ``dfs`` must have identical schema, otherwise
            exception will be thrown.
        """
        df = self.workflow.intersect(self, *dfs, distinct=distinct)
        return self._to_self_type(df)

    def distinct(self: TDF) -> TDF:
        """Get distinct dataframe. Equivalent to ``SELECT DISTINCT * FROM df``

        :return: dataframe with unique records
        """
        df = self.workflow.process(self, using=Distinct)
        return self._to_self_type(df)

    def dropna(self: TDF, how: str='any', thresh: int=None, subset: List[str]=None) -> TDF:
        """Drops records containing NA records

        :param how: 'any' or 'all'. 'any' drops rows that contain any nulls.
          'all' drops rows that contain all nulls.
        :param thresh: int, drops rows that have less than thresh non-null values
        :param subset: list of columns to operate on
        :return: dataframe with incomplete records dropped
        """
        params = dict(how=how, thresh=thresh, subset=subset)
        params = {k: v for k, v in params.items() if v is not None}
        df = self.workflow.process(self, using=Dropna, params=params)
        return self._to_self_type(df)

    def fillna(self: TDF, value: Any, subset: List[str]=None) -> TDF:
        """Fills NA values with replacement values

        :param value: if scalar, fills all columns with same value.
            if dictionary, fills NA using the keys as column names and the
            values as the replacement values.
        :param subset: list of columns to operate on. ignored if value is
            a dictionary

        :return: dataframe with NA records filled
        """
        params = dict(value=value, subset=subset)
        params = {k: v for k, v in params.items() if v is not None}
        df = self.workflow.process(self, using=Fillna, params=params)
        return self._to_self_type(df)

    def sample(self: TDF, n: Optional[int]=None, frac: Optional[float]=None, replace: bool=False, seed: Optional[int]=None) -> TDF:
        """
        Sample dataframe by number of rows or by fraction

        :param n: number of rows to sample, one and only one of ``n`` and ``fact``
          must be set
        :param frac: fraction [0,1] to sample, one and only one of ``n`` and ``fact``
          must be set
        :param replace: whether replacement is allowed. With replacement,
          there may be duplicated rows in the result, defaults to False
        :param seed: seed for randomness, defaults to None

        :return: sampled dataframe
        """
        params: Dict[str, Any] = dict(replace=replace)
        if seed is not None:
            params['seed'] = seed
        if n is not None:
            params['n'] = n
        if frac is not None:
            params['frac'] = frac
        df = self.workflow.process(self, using=Sample, params=params)
        return self._to_self_type(df)

    def take(self: TDF, n: int, presort: str=None, na_position: str='last') -> TDF:
        """
        Get the first n rows of a DataFrame per partition. If a presort is defined,
        use the presort before applying take. presort overrides partition_spec.presort

        :param n: number of rows to return
        :param presort: presort expression similar to partition presort
        :param na_position: position of null values during the presort.
            can accept ``first`` or ``last``

        :return: n rows of DataFrame per partition
        """
        params: Dict[str, Any] = dict()
        params['n'] = n
        assert_or_throw(isinstance(n, int), ValueError('n needs to be an integer'))
        assert_or_throw(na_position in ('first', 'last'), ValueError("na_position must be either 'first' or 'last'"))
        params['na_position'] = na_position
        if presort is not None:
            params['presort'] = presort
        df = self.workflow.process(self, using=Take, pre_partition=self.partition_spec, params=params)
        return self._to_self_type(df)

    def weak_checkpoint(self: TDF, lazy: bool=False, **kwargs: Any) -> TDF:
        """Cache the dataframe in memory

        :param lazy: whether it is a lazy checkpoint, defaults to False (eager)
        :param kwargs: paramteters for the underlying execution engine function
        :return: the cached dataframe

        .. note::

            Weak checkpoint in most cases is the best choice for caching a dataframe to
            avoid duplicated computation. However it does not guarantee to break up the
            the compute dependency for this dataframe, so when you have very complicated
            compute, you may encounter issues such as stack overflow. Also, weak
            checkpoint normally caches the dataframe in memory, if memory is a concern,
            then you should consider :meth:`~.strong_checkpoint`
        """
        self._task.set_checkpoint(WeakCheckpoint(lazy=lazy, **kwargs))
        return self

    def strong_checkpoint(self: TDF, storage_type: str='file', lazy: bool=False, partition: Any=None, single: bool=False, **kwargs: Any) -> TDF:
        """Cache the dataframe as a temporary file

        :param storage_type: can be either ``file`` or ``table``, defaults to ``file``
        :param lazy: whether it is a lazy checkpoint, defaults to False (eager)
        :param partition: |PartitionLikeObject|, defaults to None.
        :param single: force the output as a single file, defaults to False
        :param kwargs: paramteters for the underlying execution engine function
        :return: the cached dataframe

        .. note::

            Strong checkpoint guarantees the output dataframe compute dependency is
            from the temporary file. Use strong checkpoint only when
            :meth:`~.weak_checkpoint` can't be used.

            Strong checkpoint file will be removed after the execution of the workflow.
        """
        self._task.set_checkpoint(StrongCheckpoint(storage_type=storage_type, obj_id=str(uuid4()), deterministic=False, permanent=False, lazy=lazy, partition=partition, single=single, **kwargs))
        return self

    def deterministic_checkpoint(self: TDF, storage_type: str='file', lazy: bool=False, partition: Any=None, single: bool=False, namespace: Any=None, **kwargs: Any) -> TDF:
        """Cache the dataframe as a temporary file

        :param storage_type: can be either ``file`` or ``table``, defaults to ``file``
        :param lazy: whether it is a lazy checkpoint, defaults to False (eager)
        :param partition: |PartitionLikeObject|, defaults to None.
        :param single: force the output as a single file, defaults to False
        :param kwargs: paramteters for the underlying execution engine function
        :param namespace: a value to control determinism, defaults to None.
        :return: the cached dataframe

        .. note::

            The difference vs :meth:`~.strong_checkpoint` is that this checkpoint is not
            removed after execution, so it can take effect cross execution if the
            dependent compute logic is not changed.
        """
        self._task.set_checkpoint(StrongCheckpoint(storage_type=storage_type, obj_id=self._task.__uuid__(), deterministic=True, permanent=True, lazy=lazy, partition=partition, single=single, namespace=namespace, **kwargs))
        return self

    def yield_file_as(self: TDF, name: str) -> None:
        """Cache the dataframe in file

        :param name: the name of the yielded dataframe

        .. note::

            In only the following cases you can yield file/table:

            * you have not checkpointed (persisted) the dataframe, for example
              ``df.yield_file_as("a")``
            * you have used :meth:`~.deterministic_checkpoint`, for example
              ``df.deterministic_checkpoint().yield_file_as("a")``
            * yield is workflow, compile level logic

            For the first case, the yield will also be a strong checkpoint so
            whenever you yield a dataframe as a file, the dataframe has been saved as a
            file and loaded back as a new dataframe.
        """
        if not self._task.has_checkpoint:
            self.deterministic_checkpoint(storage_type='file', namespace=str(uuid4()))
        self.workflow._yields[name] = self._task.yielded

    def yield_table_as(self: TDF, name: str) -> None:
        """Cache the dataframe as a table

        :param name: the name of the yielded dataframe

        .. note::

            In only the following cases you can yield file/table:

            * you have not checkpointed (persisted) the dataframe, for example
              ``df.yield_file_as("a")``
            * you have used :meth:`~.deterministic_checkpoint`, for example
              ``df.deterministic_checkpoint().yield_file_as("a")``
            * yield is workflow, compile level logic

            For the first case, the yield will also be a strong checkpoint so
            whenever you yield a dataframe as a file, the dataframe has been saved as a
            file and loaded back as a new dataframe.
        """
        if not self._task.has_checkpoint:
            self.deterministic_checkpoint(storage_type='table', namespace=str(uuid4()))
        self.workflow._yields[name] = self._task.yielded

    def yield_dataframe_as(self: TDF, name: str, as_local: bool=False) -> None:
        """Yield a dataframe that can be accessed without
        the current execution engine

        :param name: the name of the yielded dataframe
        :param as_local: yield the local version of the dataframe

        .. note::

            When ``as_local`` is True, it can trigger an additional compute
            to do the conversion. To avoid recompute, you should add
            ``persist`` before yielding.
        """
        yielded = YieldedDataFrame(self._task.__uuid__())
        self.workflow._yields[name] = yielded
        self._task.set_yield_dataframe_handler(lambda df: yielded.set_value(df), as_local=as_local)

    def persist(self: TDF) -> TDF:
        """Persist the current dataframe

        :return: the persisted dataframe
        :rtype: :class:`~.WorkflowDataFrame`

        .. note::

            ``persist`` can only guarantee the persisted dataframe will be computed
            for only once. However this doesn't mean the backend really breaks up the
            execution dependency at the persisting point. Commonly, it doesn't cause
            any issue, but if your execution graph is long, it may cause expected
            problems for example, stack overflow.

            ``persist`` method is considered as weak checkpoint. Sometimes, it may be
            necessary to use strong checkpint, which is :meth:`~.checkpoint`
        """
        return self.weak_checkpoint(lazy=False)

    def checkpoint(self: TDF, storage_type: str='file') -> TDF:
        return self.strong_checkpoint(storage_type=storage_type, lazy=False)

    def broadcast(self: TDF) -> TDF:
        """Broadcast the current dataframe

        :return: the broadcasted dataframe
        :rtype: :class:`~.WorkflowDataFrame`
        """
        self._task.broadcast()
        return self

    def partition(self: TDF, *args: Any, **kwargs: Any) -> TDF:
        """Partition the current dataframe. Please read |PartitionTutorial|

        :param args: |PartitionLikeObject|
        :param kwargs: |PartitionLikeObject|
        :return: dataframe with the partition hint
        :rtype: :class:`~.WorkflowDataFrame`

        .. note::

            Normally this step is fast because it's to add a partition hint
            for the next step.
        """
        res = WorkflowDataFrame(self.workflow, self._task)
        res.reset_metadata({'pre_partition': PartitionSpec(*args, **kwargs)})
        return self._to_self_type(res)

    def partition_by(self: TDF, *keys: str, **kwargs: Any) -> TDF:
        """Partition the current dataframe by keys. Please read |PartitionTutorial|.
        This is a wrapper of :meth:`~.partition`

        :param keys: partition keys
        :param kwargs: |PartitionLikeObject| excluding ``by`` and ``partition_by``
        :return: dataframe with the partition hint
        :rtype: :class:`~.WorkflowDataFrame`
        """
        assert_or_throw(len(keys) > 0, FugueWorkflowCompileError("keys can't be empty"))
        assert_or_throw('by' not in kwargs and 'partition_by' not in kwargs, FugueWorkflowCompileError("by and partition_by can't be in kwargs"))
        return self.partition(by=keys, **kwargs)

    def per_partition_by(self: TDF, *keys: str) -> TDF:
        """Partition the current dataframe by keys so each physical partition contains
        only one logical partition. Please read |PartitionTutorial|.
        This is a wrapper of :meth:`~.partition`

        :param keys: partition keys
        :return: dataframe that is both logically and physically partitioned by ``keys``
        :rtype: :class:`~.WorkflowDataFrame`

        .. note::

            This is a hint but not enforced, certain execution engines will not
            respect this hint.
        """
        return self.partition_by(*keys, algo='even')

    def per_row(self: TDF) -> TDF:
        """Partition the current dataframe to one row per partition.
        Please read |PartitionTutorial|. This is a wrapper of :meth:`~.partition`

        :return: dataframe that is evenly partitioned by row count
        :rtype: :class:`~.WorkflowDataFrame`

        .. note::

            This is a hint but not enforced, certain execution engines will not
            respect this hint.
        """
        return self.partition('per_row')

    def _to_self_type(self: TDF, df: 'WorkflowDataFrame') -> TDF:
        return df

    def drop(self: TDF, columns: List[str], if_exists: bool=False) -> TDF:
        """Drop columns from the dataframe.

        :param columns: columns to drop
        :param if_exists: if setting to True, it will ignore non-existent columns,
          defaults to False
        :return: the dataframe after dropping columns
        :rtype: :class:`~.WorkflowDataFrame`
        """
        df = self.workflow.process(self, using=DropColumns, params=dict(columns=columns, if_exists=if_exists))
        return self._to_self_type(df)

    def rename(self: TDF, *args: Any, **kwargs: Any) -> TDF:
        """Rename the dataframe using a mapping dict

        :param args: list of dicts containing rename maps
        :param kwargs: rename map
        :return: a new dataframe with the new names
        :rtype: :class:`~.WorkflowDataFrame`

        .. note::

            This interface is more flexible than
            :meth:`fugue.dataframe.dataframe.DataFrame.rename`

        .. admonition:: Examples

            >>> df.rename({"a": "b"}, c="d", e="f")
        """
        m: Dict[str, str] = {}
        for a in args:
            m.update(a)
        m.update(kwargs)
        df = self.workflow.process(self, using=Rename, params=dict(columns=m))
        return self._to_self_type(df)

    def alter_columns(self: TDF, columns: Any) -> TDF:
        """Change column types

        :param columns: |SchemaLikeObject|
        :return: a new dataframe with the new column types
        :rtype: :class:`~.WorkflowDataFrame`

        .. note::

            The output dataframe will not change the order of original schema.

        .. admonition:: Examples

            >>> df.alter_columns("a:int,b;str")
        """
        df = self.workflow.process(self, using=AlterColumns, params=dict(columns=columns))
        return self._to_self_type(df)

    def zip(self: TDF, *dfs: Any, how: str='inner', partition: Any=None, temp_path: Optional[str]=None, to_file_threshold: Any=-1) -> TDF:
        """Zip this data frame with multiple dataframes together
        with given partition specifications. It's a wrapper of
        :meth:`fugue.workflow.workflow.FugueWorkflow.zip`.

        :param dfs: |DataFramesLikeObject|
        :param how: can accept ``inner``, ``left_outer``, ``right_outer``,
          ``full_outer``, ``cross``, defaults to ``inner``
        :param partition: |PartitionLikeObject|, defaults to None.
        :param temp_path: file path to store the data (used only if the serialized data
          is larger than ``to_file_threshold``), defaults to None
        :param to_file_threshold: file byte size threshold, defaults to -1

        :return: a zipped dataframe
        :rtype: :class:`~.WorkflowDataFrame`

        .. note::

            * ``dfs`` must be list like, the zipped dataframe will be list like
            * ``dfs`` is fine to be empty
            * If you want dict-like zip, use
              :meth:`fugue.workflow.workflow.FugueWorkflow.zip`

        .. seealso::

            Read |CoTransformer| and |ZipComap| for details
        """
        if partition is None:
            partition = self.partition_spec
        df = self.workflow.zip(self, *dfs, how=how, partition=partition, temp_path=temp_path, to_file_threshold=to_file_threshold)
        return self._to_self_type(df)

    def __getitem__(self: TDF, columns: List[Any]) -> TDF:
        df = self.workflow.process(self, using=SelectColumns, params=dict(columns=columns))
        return self._to_self_type(df)

    def save(self, path: str, fmt: str='', mode: str='overwrite', partition: Any=None, single: bool=False, **kwargs: Any) -> None:
        """Save this dataframe to a persistent storage

        :param path: output path
        :param fmt: format hint can accept ``parquet``, ``csv``, ``json``,
          defaults to None, meaning to infer
        :param mode: can accept ``overwrite``, ``append``, ``error``,
          defaults to "overwrite"
        :param partition: |PartitionLikeObject|, how to partition the
          dataframe before saving, defaults to empty
        :param single: force the output as a single file, defaults to False
        :param kwargs: parameters to pass to the underlying framework

        For more details and examples, read
        :ref:`Save & Load <tutorial:tutorials/advanced/dag:save & load>`.
        """
        if partition is None:
            partition = self.partition_spec
        self.workflow.output(self, using=Save, pre_partition=partition, params=dict(path=path, fmt=fmt, mode=mode, single=single, params=kwargs))

    def save_and_use(self: TDF, path: str, fmt: str='', mode: str='overwrite', partition: Any=None, single: bool=False, **kwargs: Any) -> TDF:
        """Save this dataframe to a persistent storage and load back to use
        in the following steps

        :param path: output path
        :param fmt: format hint can accept ``parquet``, ``csv``, ``json``,
          defaults to None, meaning to infer
        :param mode: can accept ``overwrite``, ``append``, ``error``,
          defaults to "overwrite"
        :param partition: |PartitionLikeObject|, how to partition the
          dataframe before saving, defaults to empty
        :param single: force the output as a single file, defaults to False
        :param kwargs: parameters to pass to the underlying framework

        For more details and examples, read
        :ref:`Save & Load <tutorial:tutorials/advanced/dag:save & load>`.
        """
        if partition is None:
            partition = self.partition_spec
        df = self.workflow.process(self, using=SaveAndUse, pre_partition=partition, params=dict(path=path, fmt=fmt, mode=mode, single=single, params=kwargs))
        return self._to_self_type(df)

    @property
    def schema(self) -> Schema:
        """
        :raises NotImplementedError: don't call this method
        """
        raise NotImplementedError('WorkflowDataFrame does not support this method')

    @property
    def is_local(self) -> bool:
        """
        :raises NotImplementedError: don't call this method
        """
        raise NotImplementedError('WorkflowDataFrame does not support this method')

    def as_local(self) -> DataFrame:
        """
        :raises NotImplementedError: don't call this method
        """
        raise NotImplementedError('WorkflowDataFrame does not support this method')

    def as_local_bounded(self) -> DataFrame:
        """
        :raises NotImplementedError: don't call this method
        """
        raise NotImplementedError('WorkflowDataFrame does not support this method')

    @property
    def is_bounded(self) -> bool:
        """
        :raises NotImplementedError: don't call this method
        """
        raise NotImplementedError('WorkflowDataFrame does not support this method')

    @property
    def empty(self) -> bool:
        """
        :raises NotImplementedError: don't call this method
        """
        raise NotImplementedError('WorkflowDataFrame does not support this method')

    @property
    def num_partitions(self) -> int:
        """
        :raises NotImplementedError: don't call this method
        """
        raise NotImplementedError('WorkflowDataFrame does not support this method')

    def peek_array(self) -> List[Any]:
        """
        :raises NotImplementedError: don't call this method
        """
        raise NotImplementedError('WorkflowDataFrame does not support this method')

    def count(self) -> int:
        """
        :raises NotImplementedError: don't call this method
        """
        raise NotImplementedError('WorkflowDataFrame does not support this method')

    def as_array(self, columns: Optional[List[str]]=None, type_safe: bool=False) -> List[Any]:
        """
        :raises NotImplementedError: don't call this method
        """
        raise NotImplementedError('WorkflowDataFrame does not support this method')

    def as_array_iterable(self, columns: Optional[List[str]]=None, type_safe: bool=False) -> Iterable[Any]:
        """
        :raises NotImplementedError: don't call this method
        """
        raise NotImplementedError('WorkflowDataFrame does not support this method')

    def _drop_cols(self: TDF, cols: List[str]) -> DataFrame:
        raise NotImplementedError('WorkflowDataFrame does not support this method')

    def _select_cols(self, keys: List[Any]) -> DataFrame:
        raise NotImplementedError('WorkflowDataFrame does not support this method')