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
@staticmethod
def _uses_inferred_column_names(names, skiprows, skipfooter, usecols):
    """
        Tell whether need to use inferred column names in workers or not.

        1) ``False`` is returned in 2 cases and means next:
            1.a) `names` parameter was provided from the API layer. In this case parameter
            `names` must be provided as `names` parameter for ``read_csv`` in the workers.
            1.b) `names` parameter wasn't provided from the API layer. In this case column names
            inference must happen in each partition.
        2) ``True`` is returned in case when inferred column names from pre-reading stage must be
            provided as `names` parameter for ``read_csv`` in the workers.

        In case `names` was provided, the other parameters aren't checked. Otherwise, inferred column
        names should be used in a case of not full data reading which is defined by `skipfooter` parameter,
        when need to skip lines at the bottom of file or by `skiprows` parameter, when need to skip lines at
        the top of file (but if `usecols` was provided, column names inference must happen in the workers).

        Parameters
        ----------
        names : array-like
            List of column names to use.
        skiprows : list-like, int or callable
            Line numbers to skip (0-indexed) or number of lines to skip (int) at
            the start of the file. If callable, the callable function will be
            evaluated against the row indices, returning ``True`` if the row should
            be skipped and ``False`` otherwise.
        skipfooter : int
            Number of lines at bottom of the file to skip.
        usecols : list-like or callable
            Subset of the columns.

        Returns
        -------
        bool
            Whether to use inferred column names in ``read_csv`` of the workers or not.
        """
    if names not in [None, lib.no_default]:
        return False
    if skipfooter != 0:
        return True
    if isinstance(skiprows, int) and skiprows == 0:
        return False
    if is_list_like(skiprows):
        return usecols is None
    return skiprows is not None