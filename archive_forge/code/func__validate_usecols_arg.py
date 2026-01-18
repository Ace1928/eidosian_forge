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
@_inherit_docstrings(pandas.io.parsers.base_parser.ParserBase._validate_usecols_arg)
def _validate_usecols_arg(cls, usecols):
    msg = "'usecols' must either be list-like of all strings, all unicode, " + 'all integers or a callable.'
    if usecols is not None:
        if callable(usecols):
            return (usecols, None)
        if not is_list_like(usecols):
            raise ValueError(msg)
        usecols_dtype = lib.infer_dtype(usecols, skipna=False)
        if usecols_dtype not in ('empty', 'integer', 'string'):
            raise ValueError(msg)
        usecols = set(usecols)
        return (usecols, usecols_dtype)
    return (usecols, None)