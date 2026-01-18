from __future__ import annotations
from abc import (
from collections import abc
from io import StringIO
from itertools import islice
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.json import (
from pandas._libs.tslibs import iNaT
from pandas.compat._optional import import_optional_dependency
from pandas.errors import AbstractMethodError
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
from pandas.core.reshape.concat import concat
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import (
from pandas.io.json._normalize import convert_to_line_delimits
from pandas.io.json._table_schema import (
from pandas.io.parsers.readers import validate_integer
def _get_data_from_filepath(self, filepath_or_buffer):
    """
        The function read_json accepts three input types:
            1. filepath (string-like)
            2. file-like object (e.g. open file object, StringIO)
            3. JSON string

        This method turns (1) into (2) to simplify the rest of the processing.
        It returns input types (2) and (3) unchanged.

        It raises FileNotFoundError if the input is a string ending in
        one of .json, .json.gz, .json.bz2, etc. but no such file exists.
        """
    filepath_or_buffer = stringify_path(filepath_or_buffer)
    if not isinstance(filepath_or_buffer, str) or is_url(filepath_or_buffer) or is_fsspec_url(filepath_or_buffer) or file_exists(filepath_or_buffer):
        self.handles = get_handle(filepath_or_buffer, 'r', encoding=self.encoding, compression=self.compression, storage_options=self.storage_options, errors=self.encoding_errors)
        filepath_or_buffer = self.handles.handle
    elif isinstance(filepath_or_buffer, str) and filepath_or_buffer.lower().endswith(('.json',) + tuple((f'.json{c}' for c in extension_to_compression))) and (not file_exists(filepath_or_buffer)):
        raise FileNotFoundError(f'File {filepath_or_buffer} does not exist')
    else:
        warnings.warn("Passing literal json to 'read_json' is deprecated and will be removed in a future version. To read from a literal string, wrap it in a 'StringIO' object.", FutureWarning, stacklevel=find_stack_level())
    return filepath_or_buffer