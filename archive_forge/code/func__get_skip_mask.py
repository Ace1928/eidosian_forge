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
def _get_skip_mask(cls, rows_index: pandas.Index, skiprows: Callable):
    """
        Get mask of skipped by callable `skiprows` rows.

        Parameters
        ----------
        rows_index : pandas.Index
            Rows index to get mask for.
        skiprows : Callable
            Callable to check whether row index should be skipped.

        Returns
        -------
        pandas.Index
        """
    try:
        mask = skiprows(rows_index)
        assert is_list_like(mask)
    except (ValueError, TypeError, AssertionError):
        mask = rows_index.map(skiprows)
    return mask