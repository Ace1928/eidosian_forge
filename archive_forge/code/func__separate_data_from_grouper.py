import numpy as np
import pandas
from modin.config import use_range_partitioning_groupby
from modin.core.dataframe.algebra import GroupByReduce
from modin.error_message import ErrorMessage
from modin.utils import hashable
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