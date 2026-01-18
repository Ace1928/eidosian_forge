import functools
import inspect
import os
from csv import Dialect
from typing import Callable, Dict, Sequence, Tuple, Union
import pandas
import pandas._libs.lib as lib
import pyarrow as pa
from pandas.core.dtypes.common import is_list_like
from pandas.io.common import get_handle, is_url
from pyarrow.csv import ConvertOptions, ParseOptions, ReadOptions, read_csv
from modin.core.io import BaseIO
from modin.core.io.text.text_file_dispatcher import TextFileDispatcher
from modin.error_message import ErrorMessage
from modin.experimental.core.execution.native.implementations.hdk_on_native.dataframe.dataframe import (
from modin.experimental.core.storage_formats.hdk.query_compiler import (
from modin.utils import _inherit_docstrings
@classmethod
def _read_csv_check_support(cls, read_csv_kwargs: ReadCsvKwargsType) -> Tuple[bool, str]:
    """
        Check if passed parameters are supported by current ``modin.pandas.read_csv`` implementation.

        Parameters
        ----------
        read_csv_kwargs : dict
                Parameters of read_csv function.

        Returns
        -------
        bool
            Whether passed parameters are supported or not.
        str
            Error message that should be raised if user explicitly set `engine="arrow"`.
        """
    filepath_or_buffer = read_csv_kwargs['filepath_or_buffer']
    header = read_csv_kwargs['header']
    names = read_csv_kwargs['names']
    engine = read_csv_kwargs['engine']
    skiprows = read_csv_kwargs['skiprows']
    delimiter = read_csv_kwargs['delimiter']
    parse_dates = read_csv_kwargs['parse_dates']
    if read_csv_kwargs['compression'] != 'infer':
        return (False, "read_csv with 'arrow' engine doesn't support explicit compression parameter, compression" + ' must be inferred automatically (supported compression types are gzip and bz2)')
    if isinstance(filepath_or_buffer, str):
        if not os.path.exists(filepath_or_buffer):
            if cls.file_exists(filepath_or_buffer) or is_url(filepath_or_buffer):
                return (False, "read_csv with 'arrow' engine supports only local files")
            else:
                raise FileNotFoundError('No such file or directory')
    elif not cls.pathlib_or_pypath(filepath_or_buffer):
        if hasattr(filepath_or_buffer, 'read'):
            return (False, "read_csv with 'arrow' engine doesn't support file-like objects")
        else:
            raise ValueError(f'Invalid file path or buffer object type: {type(filepath_or_buffer)}')
    if read_csv_kwargs.get('skipfooter') and read_csv_kwargs.get('nrows'):
        return (False, 'Exception is raised by pandas itself')
    for arg, def_value in cls.read_csv_unsup_defaults.items():
        if read_csv_kwargs[arg] != def_value:
            return (False, f"read_csv with 'arrow' engine doesn't support {arg} parameter")
    if delimiter is not None and read_csv_kwargs['delim_whitespace'] is True:
        raise ValueError('Specified a delimiter with both sep and delim_whitespace=True; you can only specify one.')
    parse_dates_unsupported = isinstance(parse_dates, dict) or (isinstance(parse_dates, list) and any((not isinstance(date, str) for date in parse_dates)))
    if parse_dates_unsupported:
        return (False, "read_csv with 'arrow' engine supports only bool and " + 'flattened list of string column names for the ' + "'parse_dates' parameter")
    if names and names != lib.no_default:
        if header not in [None, 0, 'infer']:
            return (False, "read_csv with 'arrow' engine and provided 'names' parameter supports only 0, None and " + "'infer' header values")
        if isinstance(parse_dates, list) and (not set(parse_dates).issubset(names)):
            missing_columns = set(parse_dates) - set(names)
            raise ValueError(f"Missing column provided to 'parse_dates': '{', '.join(missing_columns)}'")
        empty_pandas_df = pandas.read_csv(**dict(read_csv_kwargs, nrows=0, skiprows=None, skipfooter=0, usecols=None, index_col=None, names=None, parse_dates=None, engine=None if engine == 'arrow' else engine))
        columns_number = len(empty_pandas_df.columns)
        if columns_number != len(names):
            return (False, "read_csv with 'arrow' engine doesn't support names parameter, which length doesn't match " + 'with actual number of columns')
    else:
        if header not in [0, 'infer']:
            return (False, "read_csv with 'arrow' engine without 'names' parameter provided supports only 0 and 'infer' " + 'header values')
        if isinstance(parse_dates, list):
            empty_pandas_df = pandas.read_csv(**dict(read_csv_kwargs, nrows=0, skiprows=None, skipfooter=0, usecols=None, index_col=None, engine=None if engine == 'arrow' else engine))
            if not set(parse_dates).issubset(empty_pandas_df.columns):
                raise ValueError("Missing column provided to 'parse_dates'")
    if not read_csv_kwargs['skip_blank_lines']:
        return (False, "read_csv with 'arrow' engine doesn't support skip_blank_lines = False parameter")
    if skiprows is not None and (not isinstance(skiprows, int)):
        return (False, "read_csv with 'arrow' engine doesn't support non-integer skiprows parameter")
    return (True, None)