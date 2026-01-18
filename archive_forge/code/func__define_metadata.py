import codecs
import io
import os
import warnings
from csv import QUOTE_NONE
from typing import Callable, Optional, Sequence, Tuple, Union
import numpy as np
import pandas
import pandas._libs.lib as lib
from pandas.core.dtypes.common import is_list_like
from pandas.io.common import stringify_path
from modin.config import MinPartitionSize, NPartitions
from modin.core.io.file_dispatcher import FileDispatcher, OpenFile
from modin.core.io.text.utils import CustomNewlineIterator
from modin.core.storage_formats.pandas.utils import compute_chunksize
from modin.utils import _inherit_docstrings
@classmethod
def _define_metadata(cls, df: pandas.DataFrame, column_names: ColumnNamesTypes) -> Tuple[list, int]:
    """
        Define partitioning metadata.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to split.
        column_names : ColumnNamesTypes
            Column names of df.

        Returns
        -------
        column_widths : list
            Column width to use during new frame creation (number of
            columns for each partition).
        num_splits : int
            The maximum number of splits to separate the DataFrame into.
        """
    num_splits = min(len(column_names) or 1, NPartitions.get())
    min_block_size = MinPartitionSize.get()
    column_chunksize = compute_chunksize(df.shape[1], num_splits, min_block_size)
    if column_chunksize > len(column_names):
        column_widths = [len(column_names)]
        num_splits = 1
    else:
        column_widths = [column_chunksize if len(column_names) > column_chunksize * (i + 1) else 0 if len(column_names) < column_chunksize * i else len(column_names) - column_chunksize * i for i in range(num_splits)]
    return (column_widths, num_splits)