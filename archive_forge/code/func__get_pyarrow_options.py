from __future__ import annotations
from typing import TYPE_CHECKING
import warnings
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.inference import is_integer
import pandas as pd
from pandas import DataFrame
from pandas.io._util import (
from pandas.io.parsers.base_parser import ParserBase
def _get_pyarrow_options(self) -> None:
    """
        Rename some arguments to pass to pyarrow
        """
    mapping = {'usecols': 'include_columns', 'na_values': 'null_values', 'escapechar': 'escape_char', 'skip_blank_lines': 'ignore_empty_lines', 'decimal': 'decimal_point', 'quotechar': 'quote_char'}
    for pandas_name, pyarrow_name in mapping.items():
        if pandas_name in self.kwds and self.kwds.get(pandas_name) is not None:
            self.kwds[pyarrow_name] = self.kwds.pop(pandas_name)
    date_format = self.date_format
    if isinstance(date_format, str):
        date_format = [date_format]
    else:
        date_format = None
    self.kwds['timestamp_parsers'] = date_format
    self.parse_options = {option_name: option_value for option_name, option_value in self.kwds.items() if option_value is not None and option_name in ('delimiter', 'quote_char', 'escape_char', 'ignore_empty_lines')}
    on_bad_lines = self.kwds.get('on_bad_lines')
    if on_bad_lines is not None:
        if callable(on_bad_lines):
            self.parse_options['invalid_row_handler'] = on_bad_lines
        elif on_bad_lines == ParserBase.BadLineHandleMethod.ERROR:
            self.parse_options['invalid_row_handler'] = None
        elif on_bad_lines == ParserBase.BadLineHandleMethod.WARN:

            def handle_warning(invalid_row) -> str:
                warnings.warn(f'Expected {invalid_row.expected_columns} columns, but found {invalid_row.actual_columns}: {invalid_row.text}', ParserWarning, stacklevel=find_stack_level())
                return 'skip'
            self.parse_options['invalid_row_handler'] = handle_warning
        elif on_bad_lines == ParserBase.BadLineHandleMethod.SKIP:
            self.parse_options['invalid_row_handler'] = lambda _: 'skip'
    self.convert_options = {option_name: option_value for option_name, option_value in self.kwds.items() if option_value is not None and option_name in ('include_columns', 'null_values', 'true_values', 'false_values', 'decimal_point', 'timestamp_parsers')}
    self.convert_options['strings_can_be_null'] = '' in self.kwds['null_values']
    if self.header is None and 'include_columns' in self.convert_options:
        self.convert_options['include_columns'] = [f'f{n}' for n in self.convert_options['include_columns']]
    self.read_options = {'autogenerate_column_names': self.header is None, 'skip_rows': self.header if self.header is not None else self.kwds['skiprows'], 'encoding': self.encoding}