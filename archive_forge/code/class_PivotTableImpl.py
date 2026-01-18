import numpy as np
import pandas
from modin.config import use_range_partitioning_groupby
from modin.core.dataframe.algebra import GroupByReduce
from modin.error_message import ErrorMessage
from modin.utils import hashable
class PivotTableImpl:
    """Provide MapReduce, Range-Partitioning and Full-Column implementations for 'pivot_table()'."""

    @classmethod
    def map_reduce_impl(cls, qc, unique_keys, drop_column_level, pivot_kwargs):
        """Compute 'pivot_table()' using MapReduce implementation."""
        if pivot_kwargs['margins']:
            raise NotImplementedError("MapReduce 'pivot_table' implementation doesn't support 'margins=True' parameter")
        index, columns, values = (pivot_kwargs['index'], pivot_kwargs['columns'], pivot_kwargs['values'])
        aggfunc = pivot_kwargs['aggfunc']
        if not GroupbyReduceImpl.has_impl_for(aggfunc):
            raise NotImplementedError("MapReduce 'pivot_table' implementation only supports 'aggfuncs' that are implemented in 'GroupbyReduceImpl'")
        if len(set(index).intersection(columns)) > 0:
            raise NotImplementedError("MapReduce 'pivot_table' implementation doesn't support intersections of 'index' and 'columns'")
        to_group, keys_columns = cls._separate_data_from_grouper(qc, values, unique_keys)
        to_unstack = columns if index else None
        result = GroupbyReduceImpl.build_qc_method(aggfunc, finalizer_fn=lambda df: cls._pivot_table_from_groupby(df, pivot_kwargs['dropna'], drop_column_level, to_unstack, pivot_kwargs['fill_value']))(to_group, by=keys_columns, axis=0, groupby_kwargs={'observed': pivot_kwargs['observed'], 'sort': pivot_kwargs['sort']}, agg_args=(), agg_kwargs={}, drop=True)
        if to_unstack is None:
            result = result.transpose()
        return result

    @classmethod
    def full_axis_impl(cls, qc, unique_keys, drop_column_level, pivot_kwargs):
        """Compute 'pivot_table()' using full-column-axis implementation."""
        index, columns, values = (pivot_kwargs['index'], pivot_kwargs['columns'], pivot_kwargs['values'])
        to_group, keys_columns = cls._separate_data_from_grouper(qc, values, unique_keys)

        def applyier(df, other):
            """
            Build pivot table for a single partition.

            Parameters
            ----------
            df : pandas.DataFrame
                Partition of the self frame.
            other : pandas.DataFrame
                Broadcasted partition that contains `value` columns
                of the self frame.

            Returns
            -------
            pandas.DataFrame
                Pivot table for this particular partition.
            """
            concated = pandas.concat([df, other], axis=1, copy=False)
            del df, other
            result = pandas.pivot_table(concated, **pivot_kwargs)
            del concated
            if drop_column_level is not None:
                result = result.droplevel(drop_column_level, axis=1)
            if len(index) == 0 and len(columns) > 0:
                result = result.transpose()
            return result
        result = qc.__constructor__(to_group._modin_frame.broadcast_apply_full_axis(axis=0, func=applyier, other=keys_columns._modin_frame))
        if len(index) == 0 and len(columns) > 0:
            result = result.transpose()
        return result

    @classmethod
    def range_partition_impl(cls, qc, unique_keys, drop_column_level, pivot_kwargs):
        """Compute 'pivot_table()' using Range-Partitioning implementation."""
        if pivot_kwargs['margins']:
            raise NotImplementedError("Range-partitioning 'pivot_table' implementation doesn't support 'margins=True' parameter")
        index, columns, values = (pivot_kwargs['index'], pivot_kwargs['columns'], pivot_kwargs['values'])
        if len(set(index).intersection(columns)) > 0:
            raise NotImplementedError("Range-partitioning 'pivot_table' implementation doesn't support intersections of 'index' and 'columns'")
        if values is not None:
            to_take = list(np.unique(list(index) + list(columns) + list(values)))
            qc = qc.getitem_column_array(to_take, ignore_order=True)
        to_unstack = columns if index else None
        groupby_result = qc._groupby_shuffle(by=list(unique_keys), agg_func=pivot_kwargs['aggfunc'], axis=0, groupby_kwargs={'observed': pivot_kwargs['observed'], 'sort': pivot_kwargs['sort']}, agg_args=(), agg_kwargs={}, drop=True)
        result = groupby_result._modin_frame.apply_full_axis(axis=0, func=lambda df: cls._pivot_table_from_groupby(df, pivot_kwargs['dropna'], drop_column_level, to_unstack, pivot_kwargs['fill_value'], sort=pivot_kwargs['sort'] if len(unique_keys) > 1 else False))
        if to_unstack is None:
            result = result.transpose()
        return qc.__constructor__(result)

    @staticmethod
    def _pivot_table_from_groupby(df, dropna, drop_column_level, to_unstack, fill_value, sort=False):
        """
        Convert group by aggregation result to a pivot table.

        Parameters
        ----------
        df : pandas.DataFrame
            Group by aggregation result.
        dropna : bool
            Whether to drop NaN columns.
        drop_column_level : int or None
            An extra columns level to drop.
        to_unstack : list of labels or None
            Group by keys to pass to ``.unstack()``. Reperent `columns` parameter
            for ``.pivot_table()``.
        fill_value : bool
            Fill value for NaN values.
        sort : bool, default: False
            Whether to sort the result along index.

        Returns
        -------
        pandas.DataFrame
        """
        if df.index.nlevels > 1 and to_unstack is not None:
            df = df.unstack(level=to_unstack)
        if drop_column_level is not None:
            df = df.droplevel(drop_column_level, axis=1)
        if dropna:
            df = df.dropna(axis=1, how='all')
        if fill_value is not None:
            df = df.fillna(fill_value, downcast='infer')
        if sort:
            df = df.sort_index(axis=0)
        return df

    @staticmethod
    def _separate_data_from_grouper(qc, values, unique_keys):
        """
        Split `qc` for key columns to group by and values to aggregate.

        Parameters
        ----------
        qc : PandasQueryCompiler
        values : list of labels or None
            List of columns to aggregate. ``None`` means all columns except 'unique_keys'.
        unique_keys : list of labels
            List of key columns to group by.

        Returns
        -------
        to_aggregate : PandasQueryCompiler
        keys_to_group : PandasQueryCompiler
        """
        if values is None:
            to_aggregate = qc.drop(columns=unique_keys)
        else:
            to_aggregate = qc.getitem_column_array(np.unique(values), ignore_order=True)
        keys_to_group = qc.getitem_column_array(unique_keys, ignore_order=True)
        return (to_aggregate, keys_to_group)