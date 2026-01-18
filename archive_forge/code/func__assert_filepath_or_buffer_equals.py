from datetime import datetime
from io import StringIO
from pathlib import Path
import re
from shutil import get_terminal_size
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
from pandas.io.formats import printing
import pandas.io.formats.format as fmt
def _assert_filepath_or_buffer_equals(expected):
    if filepath_or_buffer_id == 'string':
        with open(filepath_or_buffer, encoding=encoding) as f:
            result = f.read()
    elif filepath_or_buffer_id == 'pathlike':
        result = filepath_or_buffer.read_text(encoding=encoding)
    elif filepath_or_buffer_id == 'buffer':
        result = filepath_or_buffer.getvalue()
    assert result == expected