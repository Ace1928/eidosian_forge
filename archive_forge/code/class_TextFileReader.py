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
class TextFileReader(abc.Iterator):
    """

    Passed dialect overrides any of the related parser options

    """

    def __init__(self, f: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str] | list, engine: CSVEngine | None=None, **kwds) -> None:
        if engine is not None:
            engine_specified = True
        else:
            engine = 'python'
            engine_specified = False
        self.engine = engine
        self._engine_specified = kwds.get('engine_specified', engine_specified)
        _validate_skipfooter(kwds)
        dialect = _extract_dialect(kwds)
        if dialect is not None:
            if engine == 'pyarrow':
                raise ValueError("The 'dialect' option is not supported with the 'pyarrow' engine")
            kwds = _merge_with_dialect_properties(dialect, kwds)
        if kwds.get('header', 'infer') == 'infer':
            kwds['header'] = 0 if kwds.get('names') is None else None
        self.orig_options = kwds
        self._currow = 0
        options = self._get_options_with_defaults(engine)
        options['storage_options'] = kwds.get('storage_options', None)
        self.chunksize = options.pop('chunksize', None)
        self.nrows = options.pop('nrows', None)
        self._check_file_or_buffer(f, engine)
        self.options, self.engine = self._clean_options(options, engine)
        if 'has_index_names' in kwds:
            self.options['has_index_names'] = kwds['has_index_names']
        self.handles: IOHandles | None = None
        self._engine = self._make_engine(f, self.engine)

    def close(self) -> None:
        if self.handles is not None:
            self.handles.close()
        self._engine.close()

    def _get_options_with_defaults(self, engine: CSVEngine) -> dict[str, Any]:
        kwds = self.orig_options
        options = {}
        default: object | None
        for argname, default in parser_defaults.items():
            value = kwds.get(argname, default)
            if engine == 'pyarrow' and argname in _pyarrow_unsupported and (value != default) and (value != getattr(value, 'value', default)):
                raise ValueError(f"The {repr(argname)} option is not supported with the 'pyarrow' engine")
            options[argname] = value
        for argname, default in _c_parser_defaults.items():
            if argname in kwds:
                value = kwds[argname]
                if engine != 'c' and value != default:
                    if 'python' in engine and argname not in _python_unsupported:
                        pass
                    elif 'pyarrow' in engine and argname not in _pyarrow_unsupported:
                        pass
                    else:
                        raise ValueError(f'The {repr(argname)} option is not supported with the {repr(engine)} engine')
            else:
                value = default
            options[argname] = value
        if engine == 'python-fwf':
            for argname, default in _fwf_defaults.items():
                options[argname] = kwds.get(argname, default)
        return options

    def _check_file_or_buffer(self, f, engine: CSVEngine) -> None:
        if is_file_like(f) and engine != 'c' and (not hasattr(f, '__iter__')):
            raise ValueError("The 'python' engine cannot iterate through this file buffer.")

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

    def __next__(self) -> DataFrame:
        try:
            return self.get_chunk()
        except StopIteration:
            self.close()
            raise

    def _make_engine(self, f: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str] | list | IO, engine: CSVEngine='c') -> ParserBase:
        mapping: dict[str, type[ParserBase]] = {'c': CParserWrapper, 'python': PythonParser, 'pyarrow': ArrowParserWrapper, 'python-fwf': FixedWidthFieldParser}
        if engine not in mapping:
            raise ValueError(f'Unknown engine: {engine} (valid options are {mapping.keys()})')
        if not isinstance(f, list):
            is_text = True
            mode = 'r'
            if engine == 'pyarrow':
                is_text = False
                mode = 'rb'
            elif engine == 'c' and self.options.get('encoding', 'utf-8') == 'utf-8' and isinstance(stringify_path(f), str):
                is_text = False
                if 'b' not in mode:
                    mode += 'b'
            self.handles = get_handle(f, mode, encoding=self.options.get('encoding', None), compression=self.options.get('compression', None), memory_map=self.options.get('memory_map', False), is_text=is_text, errors=self.options.get('encoding_errors', 'strict'), storage_options=self.options.get('storage_options', None))
            assert self.handles is not None
            f = self.handles.handle
        elif engine != 'python':
            msg = f'Invalid file path or buffer object type: {type(f)}'
            raise ValueError(msg)
        try:
            return mapping[engine](f, **self.options)
        except Exception:
            if self.handles is not None:
                self.handles.close()
            raise

    def _failover_to_python(self) -> None:
        raise AbstractMethodError(self)

    def read(self, nrows: int | None=None) -> DataFrame:
        if self.engine == 'pyarrow':
            try:
                df = self._engine.read()
            except Exception:
                self.close()
                raise
        else:
            nrows = validate_integer('nrows', nrows)
            try:
                index, columns, col_dict = self._engine.read(nrows)
            except Exception:
                self.close()
                raise
            if index is None:
                if col_dict:
                    new_rows = len(next(iter(col_dict.values())))
                    index = RangeIndex(self._currow, self._currow + new_rows)
                else:
                    new_rows = 0
            else:
                new_rows = len(index)
            if hasattr(self, 'orig_options'):
                dtype_arg = self.orig_options.get('dtype', None)
            else:
                dtype_arg = None
            if isinstance(dtype_arg, dict):
                dtype = defaultdict(lambda: None)
                dtype.update(dtype_arg)
            elif dtype_arg is not None and pandas_dtype(dtype_arg) in (np.str_, np.object_):
                dtype = defaultdict(lambda: dtype_arg)
            else:
                dtype = None
            if dtype is not None:
                new_col_dict = {}
                for k, v in col_dict.items():
                    d = dtype[k] if pandas_dtype(dtype[k]) in (np.str_, np.object_) else None
                    new_col_dict[k] = Series(v, index=index, dtype=d, copy=False)
            else:
                new_col_dict = col_dict
            df = DataFrame(new_col_dict, columns=columns, index=index, copy=not using_copy_on_write())
            self._currow += new_rows
        return df

    def get_chunk(self, size: int | None=None) -> DataFrame:
        if size is None:
            size = self.chunksize
        if self.nrows is not None:
            if self._currow >= self.nrows:
                raise StopIteration
            size = min(size, self.nrows - self._currow)
        return self.read(nrows=size)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        self.close()