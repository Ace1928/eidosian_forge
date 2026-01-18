import csv
import functools
import itertools
import math
import os
import re
from io import BytesIO
from pathlib import Path
from string import ascii_letters
from typing import Union
import numpy as np
import pandas
import psutil
import pytest
from pandas.core.dtypes.common import (
import modin.pandas as pd
from modin.config import (
from modin.pandas.io import to_pandas
from modin.pandas.testing import (
from modin.utils import try_cast_to_pandas
def _make_csv_file(data_dir):

    def _csv_file_maker(filename=None, row_size=NROWS, force=True, delimiter=',', encoding=None, compression='infer', additional_col_values=None, remove_randomness=False, add_blank_lines=False, add_bad_lines=False, add_nan_lines=False, thousands_separator=None, decimal_separator=None, comment_col_char=None, quoting=csv.QUOTE_MINIMAL, quotechar='"', doublequote=True, escapechar=None, lineterminator=None):
        if filename is None:
            filename = get_unique_filename(data_dir=data_dir)
        if os.path.exists(filename) and (not force):
            return None
        else:
            df = generate_dataframe(row_size, additional_col_values)
            if remove_randomness:
                df = df[['col1', 'col2', 'col3', 'col4']]
            if add_nan_lines:
                for i in range(0, row_size, row_size // (row_size // 10)):
                    df.loc[i] = pandas.Series()
            if comment_col_char:
                char = comment_col_char if isinstance(comment_col_char, str) else '#'
                df.insert(loc=0, column='col_with_comments', value=[char if x + 2 == 0 else x for x in range(row_size)])
            if thousands_separator is not None:
                for col_id in ['col1', 'col3']:
                    df[col_id] = df[col_id].apply(lambda x: f'{x:,d}'.replace(',', thousands_separator))
                df['col6'] = df['col6'].apply(lambda x: f'{x:,f}'.replace(',', thousands_separator))
            filename = f'{filename}.{COMP_TO_EXT[compression]}' if compression != 'infer' else filename
            df.to_csv(filename, sep=delimiter, encoding=encoding, compression=compression, index=False, decimal=decimal_separator if decimal_separator else '.', lineterminator=lineterminator, quoting=quoting, quotechar=quotechar, doublequote=doublequote, escapechar=escapechar)
            csv_reader_writer_params = {'delimiter': delimiter, 'doublequote': doublequote, 'escapechar': escapechar, 'lineterminator': lineterminator if lineterminator else os.linesep, 'quotechar': quotechar, 'quoting': quoting}
            if add_blank_lines:
                insert_lines_to_csv(csv_name=filename, lines_positions=[x for x in range(5, row_size, row_size // (row_size // 10))], lines_type='blank', encoding=encoding, **csv_reader_writer_params)
            if add_bad_lines:
                insert_lines_to_csv(csv_name=filename, lines_positions=[x for x in range(6, row_size, row_size // (row_size // 10))], lines_type='bad', encoding=encoding, **csv_reader_writer_params)
            return filename
    return _csv_file_maker