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
class FugueWorkflow:
    """Fugue Workflow, also known as the Fugue Programming Interface.

    In Fugue, we use DAG to represent workflows, DAG construction and execution
    are different steps, this class is mainly used in the construction step, so all
    things you added to the workflow is **description** and they are not executed
    until you call :meth:`~.run`

    Read
    :ref:`this <tutorial:tutorials/advanced/dag:initialize a workflow>`
    to learn how to initialize it in different ways and pros and cons.
    """

    def __init__(self, compile_conf: Any=None):
        assert_or_throw(compile_conf is None or isinstance(compile_conf, (dict, ParamDict)), ValueError(f'FugueWorkflow no longer takes {type(compile_conf)} as the input'))
        self._lock = SerializableRLock()
        self._spec = WorkflowSpec()
        self._computed = False
        self._graph = _Graph()
        self._yields: Dict[str, Yielded] = {}
        self._compile_conf = ParamDict({**_FUGUE_GLOBAL_CONF, **ParamDict(compile_conf)})
        self._last_df: Optional[WorkflowDataFrame] = None

    @property
    def conf(self) -> ParamDict:
        """Compile time configs"""
        return self._compile_conf

    def spec_uuid(self) -> str:
        """UUID of the workflow spec (`description`)"""
        return self._spec.__uuid__()

    def run(self, engine: Any=None, conf: Any=None, **kwargs: Any) -> FugueWorkflowResult:
        """Execute the workflow and compute all dataframes.

        .. note::

            For inputs, please read
            :func:`~.fugue.api.engine_context`

        :param engine: object that can be recognized as an engine, defaults to None
        :param conf: engine config, defaults to None
        :param kwargs: additional parameters to initialize the execution engine
        :return: the result set

        .. admonition:: Examples

            .. code-block:: python

                dag = FugueWorkflow()
                df1 = dag.df([[0]],"a:int").transform(a_transformer)
                df2 = dag.df([[0]],"b:int")

                dag.run(SparkExecutionEngine)
                df1.result.show()
                df2.result.show()

                dag = FugueWorkflow()
                df1 = dag.df([[0]],"a:int").transform(a_transformer)
                df1.yield_dataframe_as("x")

                result = dag.run(SparkExecutionEngine)
                result["x"]  # SparkDataFrame

        Read
        :ref:`this <tutorial:tutorials/advanced/dag:initialize a workflow>`
        to learn how to run in different ways and pros and cons.
        """
        with self._lock:
            with engine_context(engine, engine_conf=conf) as e:
                self._computed = False
                self._workflow_ctx = FugueWorkflowContext(engine=e, compile_conf=self.conf)
                try:
                    self._workflow_ctx.run(self._spec, {})
                except Exception as ex:
                    if not self.conf.get_or_throw(FUGUE_CONF_WORKFLOW_EXCEPTION_OPTIMIZE, bool) or sys.version_info < (3, 7):
                        raise
                    conf = self.conf.get_or_throw(FUGUE_CONF_WORKFLOW_EXCEPTION_HIDE, str)
                    pre = [p for p in conf.split(',') if p != '']
                    if len(pre) == 0:
                        raise
                    ctb = modify_traceback(sys.exc_info()[2], lambda x: any((x.lower().startswith(xx) for xx in pre)))
                    if ctb is None:
                        raise
                    raise ex.with_traceback(ctb)
                self._computed = True
        return FugueWorkflowResult(self.yields)

    @property
    def yields(self) -> Dict[str, Yielded]:
        return self._yields

    @property
    def last_df(self) -> Optional[WorkflowDataFrame]:
        return self._last_df

    def __enter__(self):
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return

    def get_result(self, df: WorkflowDataFrame) -> DataFrame:
        """After :meth:`~.run`, get the result of a dataframe defined in the dag

        :return: a calculated dataframe

        .. admonition:: Examples

            .. code-block:: python

                dag = FugueWorkflow()
                df1 = dag.df([[0]],"a:int")
                dag.run()
                dag.get_result(df1).show()
        """
        assert_or_throw(self._computed, FugueWorkflowError('not computed'))
        return self._workflow_ctx.get_result(id(df._task))

    def create(self, using: Any, schema: Any=None, params: Any=None) -> WorkflowDataFrame:
        """Run a creator to create a dataframe.

        Please read the
        :doc:`Creator Tutorial <tutorial:tutorials/extensions/creator>`

        :param using: creator-like object, if it is a string, then it must be
          the alias of a registered creator
        :param schema: |SchemaLikeObject|, defaults to None. The creator
          will be able to access this value from
          :meth:`~fugue.extensions.context.ExtensionContext.output_schema`
        :param params: |ParamsLikeObject| to run the creator,
          defaults to None. The creator will be able to access this value from
          :meth:`~fugue.extensions.context.ExtensionContext.params`
        :param pre_partition: |PartitionLikeObject|, defaults to None.
          The creator will be able to access this value from
          :meth:`~fugue.extensions.context.ExtensionContext.partition_spec`
        :return: result dataframe
        """
        task = Create(creator=CreateData(using) if is_df(using) or isinstance(using, Yielded) else using, schema=schema, params=params)
        res = self.add(task)
        self._last_df = res
        return res

    def process(self, *dfs: Any, using: Any, schema: Any=None, params: Any=None, pre_partition: Any=None) -> WorkflowDataFrame:
        """Run a processor on the dataframes.

        Please read the
        :doc:`Processor Tutorial <tutorial:tutorials/extensions/processor>`

        :param dfs: |DataFramesLikeObject|
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
        """
        _dfs = self._to_dfs(*dfs)
        task = Process(len(_dfs), processor=using, schema=schema, params=params, pre_partition=pre_partition, input_names=None if not _dfs.has_key else list(_dfs.keys()))
        if _dfs.has_key:
            res = self.add(task, **_dfs)
        else:
            res = self.add(task, *_dfs.values())
        self._last_df = res
        return res

    def output(self, *dfs: Any, using: Any, params: Any=None, pre_partition: Any=None) -> None:
        """Run a outputter on dataframes.

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
        _dfs = self._to_dfs(*dfs)
        task = Output(len(_dfs), outputter=using, params=params, pre_partition=pre_partition, input_names=None if not _dfs.has_key else list(_dfs.keys()))
        if _dfs.has_key:
            self.add(task, **_dfs)
        else:
            self.add(task, *_dfs.values())

    def create_data(self, data: Any, schema: Any=None, data_determiner: Optional[Callable[[Any], Any]]=None) -> WorkflowDataFrame:
        """Create dataframe.

        :param data: |DataFrameLikeObject| or :class:`~fugue.workflow.yielded.Yielded`
        :param schema: |SchemaLikeObject|, defaults to None
        :param data_determiner: a function to compute unique id from ``data``
        :return: a dataframe of the current workflow

        .. note::

            By default, the input ``data`` does not affect the determinism of the
            workflow (but ``schema`` and ``etadata`` do), because the amount of compute
            can be unpredictable. But if you want ``data`` to affect the
            determinism of the workflow, you can provide the function to compute the
            unique id of ``data`` using ``data_determiner``
        """
        if isinstance(data, WorkflowDataFrame):
            assert_or_throw(data.workflow is self, lambda: FugueWorkflowCompileError(f'{data} does not belong to this workflow'))
            assert_or_throw(schema is None, FugueWorkflowCompileError('schema must be None when data is WorkflowDataFrame'))
            self._last_df = data
            return data
        if isinstance(data, (List, Iterable)) and (not isinstance(data, str)) or isinstance(data, Yielded) or is_df(data):
            return self.create(using=CreateData(data, schema=schema, data_determiner=data_determiner))
        raise FugueWorkflowCompileError(f"Input data of type {type(data)} can't be converted to WorkflowDataFrame")

    def df(self, data: Any, schema: Any=None, data_determiner: Optional[Callable[[Any], str]]=None) -> WorkflowDataFrame:
        """Create dataframe. Alias of :meth:`~.create_data`

        :param data: |DataFrameLikeObject| or :class:`~fugue.workflow.yielded.Yielded`
        :param schema: |SchemaLikeObject|, defaults to None
        :param data_determiner: a function to compute unique id from ``data``
        :return: a dataframe of the current workflow

        .. note::

            By default, the input ``data`` does not affect the determinism of the
            workflow (but ``schema`` and ``etadata`` do), because the amount of
            compute can be unpredictable. But if you want ``data`` to affect the
            determinism of the workflow, you can provide the function to compute
            the unique id of ``data`` using ``data_determiner``
        """
        return self.create_data(data=data, schema=schema, data_determiner=data_determiner)

    def load(self, path: str, fmt: str='', columns: Any=None, **kwargs: Any) -> WorkflowDataFrame:
        """Load dataframe from persistent storage.
        Read :ref:`this <tutorial:tutorials/advanced/dag:save & load>`
        for details.

        :param path: file path
        :param fmt: format hint can accept ``parquet``, ``csv``, ``json``,
          defaults to "", meaning to infer
        :param columns: list of columns or a |SchemaLikeObject|, defaults to None
        :return: dataframe from the file
        :rtype: WorkflowDataFrame
        """
        return self.create(using=Load, params=dict(path=path, fmt=fmt, columns=columns, params=kwargs))

    def show(self, *dfs: Any, n: int=10, with_count: bool=False, title: Optional[str]=None) -> None:
        """Show the dataframes.
        See
        :ref:`examples <tutorial:tutorials/advanced/dag:initialize a workflow>`.

        :param dfs: |DataFramesLikeObject|
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
              need to :meth:`~.WorkflowDataFrame.persist` the dataframe. Or you can
              turn on |AutoPersist|
        """
        self.output(*dfs, using=Show, params=dict(n=n, with_count=with_count, title=title))

    def join(self, *dfs: Any, how: str, on: Optional[Iterable[str]]=None) -> WorkflowDataFrame:
        """Join dataframes.
        |ReadJoin|

        :param dfs: |DataFramesLikeObject|
        :param how: can accept ``semi``, ``left_semi``, ``anti``, ``left_anti``,
          ``inner``, ``left_outer``, ``right_outer``, ``full_outer``, ``cross``
        :param on: it can always be inferred, but if you provide, it will be
          validated against the inferred keys. Default to None
        :return: joined dataframe
        """
        _on: List[str] = list(on) if on is not None else []
        return self.process(*dfs, using=RunJoin, params=dict(how=how, on=_on))

    def set_op(self, how: str, *dfs: Any, distinct: bool=True) -> WorkflowDataFrame:
        """Union, subtract or intersect dataframes.

        :param how: can accept ``union``, ``left_semi``, ``anti``, ``left_anti``,
          ``inner``, ``left_outer``, ``right_outer``, ``full_outer``, ``cross``
        :param dfs: |DataFramesLikeObject|
        :param distinct: whether to perform `distinct` after the set operation,
          default to True
        :return: result dataframe of the set operation

        .. note::

            Currently, all dataframes in ``dfs`` must have identical schema, otherwise
            exception will be thrown.
        """
        return self.process(*dfs, using=RunSetOperation, params=dict(how=how, distinct=distinct))

    def union(self, *dfs: Any, distinct: bool=True) -> WorkflowDataFrame:
        """Union dataframes in ``dfs``.

        :param dfs: |DataFramesLikeObject|
        :param distinct: whether to perform `distinct` after union,
          default to True
        :return: unioned dataframe

        .. note::

            Currently, all dataframes in ``dfs`` must have identical schema, otherwise
            exception will be thrown.
        """
        return self.set_op('union', *dfs, distinct=distinct)

    def subtract(self, *dfs: Any, distinct: bool=True) -> WorkflowDataFrame:
        """Subtract ``dfs[1:]`` from ``dfs[0]``.

        :param dfs: |DataFramesLikeObject|
        :param distinct: whether to perform `distinct` after subtraction,
          default to True
        :return: subtracted dataframe

        .. note::

            Currently, all dataframes in ``dfs`` must have identical schema, otherwise
            exception will be thrown.
        """
        return self.set_op('subtract', *dfs, distinct=distinct)

    def intersect(self, *dfs: Any, distinct: bool=True) -> WorkflowDataFrame:
        """Intersect dataframes in ``dfs``.

        :param dfs: |DataFramesLikeObject|
        :param distinct: whether to perform `distinct` after intersection,
          default to True
        :return: intersected dataframe

        .. note::

            Currently, all dataframes in ``dfs`` must have identical schema, otherwise
            exception will be thrown.
        """
        return self.set_op('intersect', *dfs, distinct=distinct)

    def zip(self, *dfs: Any, how: str='inner', partition: Any=None, temp_path: Optional[str]=None, to_file_threshold: Any=-1) -> WorkflowDataFrame:
        """Zip multiple dataframes together with given partition
        specifications.

        :param dfs: |DataFramesLikeObject|
        :param how: can accept ``inner``, ``left_outer``, ``right_outer``,
          ``full_outer``, ``cross``, defaults to ``inner``
        :param partition: |PartitionLikeObject|, defaults to None.
        :param temp_path: file path to store the data (used only if the serialized data
          is larger than ``to_file_threshold``), defaults to None
        :param to_file_threshold: file byte size threshold, defaults to -1

        :return: a zipped dataframe

        .. note::

            * If ``dfs`` is dict like, the zipped dataframe will be dict like,
              If ``dfs`` is list like, the zipped dataframe will be list like
            * It's fine to contain only one dataframe in ``dfs``

        .. seealso::

            Read |CoTransformer| and |ZipComap| for details
        """
        return self.process(*dfs, using=Zip, params=dict(how=how, temp_path=temp_path, to_file_threshold=to_file_threshold), pre_partition=partition)

    def transform(self, *dfs: Any, using: Any, schema: Any=None, params: Any=None, pre_partition: Any=None, ignore_errors: List[Any]=_DEFAULT_IGNORE_ERRORS, callback: Any=None) -> WorkflowDataFrame:
        """Transform dataframes using transformer.

        Please read |TransformerTutorial|

        :param dfs: |DataFramesLikeObject|
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

        .. note::

            :meth:`~.transform` can be lazy and will return the transformed dataframe,
            :meth:`~.out_transform` is guaranteed to execute immediately (eager) and
            return nothing
        """
        assert_or_throw(len(dfs) == 1, NotImplementedError('transform supports only single dataframe'))
        tf = _to_transformer(using, schema)
        tf._partition_spec = PartitionSpec(pre_partition)
        callback = to_rpc_handler(callback)
        tf._has_rpc_client = not isinstance(callback, EmptyRPCHandler)
        tf.validate_on_compile()
        return self.process(*dfs, using=RunTransformer, schema=None, params=dict(transformer=tf, ignore_errors=ignore_errors, params=params, rpc_handler=callback), pre_partition=pre_partition)

    def out_transform(self, *dfs: Any, using: Any, params: Any=None, pre_partition: Any=None, ignore_errors: List[Any]=_DEFAULT_IGNORE_ERRORS, callback: Any=None) -> None:
        """Transform dataframes using transformer, it materializes the execution
        immediately and returns nothing

        Please read |TransformerTutorial|

        :param dfs: |DataFramesLikeObject|
        :param using: transformer-like object, if it is a string, then it must be
          the alias of a registered output transformer/cotransformer
        :param schema: |SchemaLikeObject|, defaults to None. The transformer
          will be able to access this value from
          :meth:`~fugue.extensions.context.ExtensionContext.output_schema`
        :param params: |ParamsLikeObject| to run the processor, defaults to None.
          The transformer will be able to access this value from
          :meth:`~fugue.extensions.context.ExtensionContext.params`
        :param pre_partition: |PartitionLikeObject|, defaults to None. It's
          recommended to use the equivalent wayt, which is to call
          :meth:`~.partition` and then call :meth:`~.out_transform` without this
          parameter
        :param ignore_errors: list of exception types the transformer can ignore,
          defaults to empty list
        :param callback: |RPCHandlerLikeObject|, defaults to None

        .. note::

            :meth:`~.transform` can be lazy and will return the transformed dataframe,
            :meth:`~.out_transform` is guaranteed to execute immediately (eager) and
            return nothing
        """
        assert_or_throw(len(dfs) == 1, NotImplementedError('output transform supports only single dataframe'))
        tf = _to_output_transformer(using)
        tf._partition_spec = PartitionSpec(pre_partition)
        callback = to_rpc_handler(callback)
        tf._has_rpc_client = not isinstance(callback, EmptyRPCHandler)
        tf.validate_on_compile()
        self.output(*dfs, using=RunOutputTransformer, params=dict(transformer=tf, ignore_errors=ignore_errors, params=params, rpc_handler=callback), pre_partition=pre_partition)

    def select(self, *statements: Any, sql_engine: Any=None, sql_engine_params: Any=None, dialect: Optional[str]=FUGUE_SQL_DEFAULT_DIALECT) -> WorkflowDataFrame:
        """Execute ``SELECT`` statement using
        :class:`~fugue.execution.execution_engine.SQLEngine`

        :param statements: a list of sub-statements in string
          or :class:`~.WorkflowDataFrame`
        :param sql_engine: it can be empty string or null (use the default SQL
          engine), a string (use the registered SQL engine), an
          :class:`~fugue.execution.execution_engine.SQLEngine` type, or
          the :class:`~fugue.execution.execution_engine.SQLEngine` instance
          (you can use ``None`` to use the default one), defaults to None
        :return: result of the ``SELECT`` statement

        .. admonition:: Examples

            .. code-block:: python

                with FugueWorkflow() as dag:
                    a = dag.df([[0,"a"]],a:int,b:str)
                    b = dag.df([[0]],a:int)
                    c = dag.select("SELECT a FROM",a,"UNION SELECT * FROM",b)
                dag.run()

        Please read :ref:`this <tutorial:tutorials/advanced/dag:select query>`
        for more examples
        """
        sql: List[Tuple[bool, str]] = []
        dfs: Dict[str, DataFrame] = {}
        for s in statements:
            if isinstance(s, str):
                sql.append((False, s))
            else:
                ws = self.df(s)
                dfs[ws.name] = ws
                sql.append((True, ws.name))
        if sql[0][0]:
            sql.insert(0, (False, 'SELECT'))
        else:
            start = sql[0][1].strip()
            if not start[:10].upper().startswith('SELECT') and (not start[:10].upper().startswith('WITH')):
                sql[0] = (False, 'SELECT ' + start)
        return self.process(dfs, using=RunSQLSelect, params=dict(statement=StructuredRawSQL(sql, dialect=dialect), sql_engine=sql_engine, sql_engine_params=ParamDict(sql_engine_params)))

    def assert_eq(self, *dfs: Any, **params: Any) -> None:
        """Compare if these dataframes are equal. It's for internal, unit test
        purpose only. It will convert both dataframes to
        :class:`~fugue.dataframe.dataframe.LocalBoundedDataFrame`, so it assumes
        all dataframes are small and fast enough to convert. DO NOT use it
        on critical or expensive tasks.

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
        self.output(*dfs, using=AssertEqual, params=params)

    def assert_not_eq(self, *dfs: Any, **params: Any) -> None:
        """Assert if all dataframes are not equal to the first dataframe.
        It's for internal, unit test purpose only. It will convert both dataframes to
        :class:`~fugue.dataframe.dataframe.LocalBoundedDataFrame`, so it assumes
        all dataframes are small and fast enough to convert. DO NOT use it
        on critical or expensive tasks.

        :param dfs: |DataFramesLikeObject|
        :param digits: precision on float number comparison, defaults to 8
        :param check_order: if to compare the row orders, defaults to False
        :param check_schema: if compare schemas, defaults to True
        :param check_content: if to compare the row values, defaults to True
        :param no_pandas: if true, it will compare the string representations of the
          dataframes, otherwise, it will convert both to pandas dataframe to compare,
          defaults to False

        :raises AssertionError: if any dataframe equals to the first dataframe
        """
        self.output(*dfs, using=AssertNotEqual, params=params)

    def add(self, task: FugueTask, *args: Any, **kwargs: Any) -> WorkflowDataFrame:
        """This method should not be called directly by users. Use
        :meth:`~.create`, :meth:`~.process`, :meth:`~.output` instead
        """
        if self.conf.get_or_throw(FUGUE_CONF_WORKFLOW_EXCEPTION_OPTIMIZE, bool) and sys.version_info >= (3, 7):
            dep = self.conf.get_or_throw(FUGUE_CONF_WORKFLOW_EXCEPTION_INJECT, int)
            if dep > 0:
                conf = self.conf.get_or_throw(FUGUE_CONF_WORKFLOW_EXCEPTION_HIDE, str)
                pre = [p for p in conf.split(',') if p != '']
                task.reset_traceback(limit=dep, should_prune=lambda x: any((x.lower().startswith(xx) for xx in pre)))
        assert_or_throw(task._node_spec is None, lambda: f"can't reuse {task}")
        dep = _Dependencies(self, task, {}, *args, **kwargs)
        name = '_' + str(len(self._spec.tasks))
        wt = self._spec.add_task(name, task, dep.dependency)
        for v in dep.dependency.values():
            v = v.split('.')[0]
            self._graph.add(name, v)
            if len(self._graph.down[v]) > 1 and self.conf.get_or_throw(FUGUE_CONF_WORKFLOW_AUTO_PERSIST, bool):
                self._spec.tasks[v].set_checkpoint(WeakCheckpoint(lazy=False, level=self.conf.get_or_none(FUGUE_CONF_WORKFLOW_AUTO_PERSIST_VALUE, object)))
        return WorkflowDataFrame(self, wt)

    def _to_dfs(self, *args: Any, **kwargs: Any) -> DataFrames:
        return DataFrames(*args, **kwargs).convert(self.create_data)

    def __getattr__(self, name: str) -> Any:
        """The dummy method to avoid PyLint complaint"""
        raise AttributeError(name)