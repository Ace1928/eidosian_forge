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
def _get_new_qc(cls, partition_ids: list, index_ids: list, dtypes_ids: list, index_col: IndexColType, index_name: str, column_widths: list, column_names: ColumnNamesTypes, skiprows_md: Union[Sequence, callable, None]=None, header_size: int=None, **kwargs):
    """
        Get new query compiler from data received from workers.

        Parameters
        ----------
        partition_ids : list
            Array with references to the partitions data.
        index_ids : list
            Array with references to the partitions index objects.
        dtypes_ids : list
            Array with references to the partitions dtypes objects.
        index_col : IndexColType
            `index_col` parameter of `read_csv` function.
        index_name : str
            Name that should be assigned to the index if `index_col`
            is not provided.
        column_widths : list
            Number of columns in each partition.
        column_names : ColumnNamesTypes
            Array with columns names.
        skiprows_md : array-like or callable, optional
            Specifies rows to skip.
        header_size : int, default: 0
            Number of rows, that occupied by header.
        **kwargs : dict
            Parameters of `read_csv` function needed for postprocessing.

        Returns
        -------
        new_query_compiler : BaseQueryCompiler
            New query compiler, created from `new_frame`.
        """
    partition_ids = cls.build_partition(partition_ids, [None] * len(index_ids), column_widths)
    new_frame = cls.frame_cls(partition_ids, lambda: cls._define_index(index_ids, index_name), column_names, None, column_widths, dtypes=lambda: cls.get_dtypes(dtypes_ids, column_names))
    new_query_compiler = cls.query_compiler_cls(new_frame)
    skipfooter = kwargs.get('skipfooter', None)
    if skipfooter:
        new_query_compiler = new_query_compiler.drop(new_query_compiler.index[-skipfooter:])
    if skiprows_md is not None:
        nrows = kwargs.get('nrows', None)
        index_range = pandas.RangeIndex(len(new_query_compiler.index))
        if is_list_like(skiprows_md):
            new_query_compiler = new_query_compiler.take_2d_positional(index=index_range.delete(skiprows_md))
        elif callable(skiprows_md):
            skip_mask = cls._get_skip_mask(index_range, skiprows_md)
            if not isinstance(skip_mask, np.ndarray):
                skip_mask = skip_mask.to_numpy('bool')
            view_idx = index_range[~skip_mask]
            new_query_compiler = new_query_compiler.take_2d_positional(index=view_idx)
        else:
            raise TypeError(f'Not acceptable type of `skiprows` parameter: {type(skiprows_md)}')
        if not isinstance(new_query_compiler.index, pandas.MultiIndex):
            new_query_compiler = new_query_compiler.reset_index(drop=True)
        if nrows:
            new_query_compiler = new_query_compiler.take_2d_positional(pandas.RangeIndex(len(new_query_compiler.index))[:nrows])
    if index_col is None or index_col is False:
        new_query_compiler._modin_frame.synchronize_labels(axis=0)
    return new_query_compiler