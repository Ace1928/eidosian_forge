from __future__ import annotations
from collections import (
import csv
import sys
from textwrap import fill
from typing import (
import warnings
import numpy as np
from pandas._config import using_copy_on_write
from pandas._libs import lib
from pandas._libs.parsers import STR_NA_VALUES
from pandas.errors import (
from pandas.util._decorators import Appender
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import (
from pandas import Series
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import RangeIndex
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import (
from pandas.io.parsers.arrow_parser_wrapper import ArrowParserWrapper
from pandas.io.parsers.base_parser import (
from pandas.io.parsers.c_parser_wrapper import CParserWrapper
from pandas.io.parsers.python_parser import (
def _clean_options(self, options: dict[str, Any], engine: CSVEngine) -> tuple[dict[str, Any], CSVEngine]:
    result = options.copy()
    fallback_reason = None
    if engine == 'c':
        if options['skipfooter'] > 0:
            fallback_reason = "the 'c' engine does not support skipfooter"
            engine = 'python'
    sep = options['delimiter']
    delim_whitespace = options['delim_whitespace']
    if sep is None and (not delim_whitespace):
        if engine in ('c', 'pyarrow'):
            fallback_reason = f"the '{engine}' engine does not support sep=None with delim_whitespace=False"
            engine = 'python'
    elif sep is not None and len(sep) > 1:
        if engine == 'c' and sep == '\\s+':
            result['delim_whitespace'] = True
            del result['delimiter']
        elif engine not in ('python', 'python-fwf'):
            fallback_reason = f"the '{engine}' engine does not support regex separators (separators > 1 char and different from '\\s+' are interpreted as regex)"
            engine = 'python'
    elif delim_whitespace:
        if 'python' in engine:
            result['delimiter'] = '\\s+'
    elif sep is not None:
        encodeable = True
        encoding = sys.getfilesystemencoding() or 'utf-8'
        try:
            if len(sep.encode(encoding)) > 1:
                encodeable = False
        except UnicodeDecodeError:
            encodeable = False
        if not encodeable and engine not in ('python', 'python-fwf'):
            fallback_reason = f"the separator encoded in {encoding} is > 1 char long, and the '{engine}' engine does not support such separators"
            engine = 'python'
    quotechar = options['quotechar']
    if quotechar is not None and isinstance(quotechar, (str, bytes)):
        if len(quotechar) == 1 and ord(quotechar) > 127 and (engine not in ('python', 'python-fwf')):
            fallback_reason = f"ord(quotechar) > 127, meaning the quotechar is larger than one byte, and the '{engine}' engine does not support such quotechars"
            engine = 'python'
    if fallback_reason and self._engine_specified:
        raise ValueError(fallback_reason)
    if engine == 'c':
        for arg in _c_unsupported:
            del result[arg]
    if 'python' in engine:
        for arg in _python_unsupported:
            if fallback_reason and result[arg] != _c_parser_defaults.get(arg):
                raise ValueError(f"Falling back to the 'python' engine because {fallback_reason}, but this causes {repr(arg)} to be ignored as it is not supported by the 'python' engine.")
            del result[arg]
    if fallback_reason:
        warnings.warn(f"Falling back to the 'python' engine because {fallback_reason}; you can avoid this warning by specifying engine='python'.", ParserWarning, stacklevel=find_stack_level())
    index_col = options['index_col']
    names = options['names']
    converters = options['converters']
    na_values = options['na_values']
    skiprows = options['skiprows']
    validate_header_arg(options['header'])
    if index_col is True:
        raise ValueError("The value of index_col couldn't be 'True'")
    if is_index_col(index_col):
        if not isinstance(index_col, (list, tuple, np.ndarray)):
            index_col = [index_col]
    result['index_col'] = index_col
    names = list(names) if names is not None else names
    if converters is not None:
        if not isinstance(converters, dict):
            raise TypeError(f'Type converters must be a dict or subclass, input was a {type(converters).__name__}')
    else:
        converters = {}
    keep_default_na = options['keep_default_na']
    floatify = engine != 'pyarrow'
    na_values, na_fvalues = _clean_na_values(na_values, keep_default_na, floatify=floatify)
    if engine == 'pyarrow':
        if not is_integer(skiprows) and skiprows is not None:
            raise ValueError("skiprows argument must be an integer when using engine='pyarrow'")
    else:
        if is_integer(skiprows):
            skiprows = list(range(skiprows))
        if skiprows is None:
            skiprows = set()
        elif not callable(skiprows):
            skiprows = set(skiprows)
    result['names'] = names
    result['converters'] = converters
    result['na_values'] = na_values
    result['na_fvalues'] = na_fvalues
    result['skiprows'] = skiprows
    return (result, engine)