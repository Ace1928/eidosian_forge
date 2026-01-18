from __future__ import annotations
from datetime import (
from functools import partial
from io import BytesIO
import os
from pathlib import Path
import platform
import re
from urllib.error import URLError
from zipfile import BadZipFile
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def _is_valid_engine_ext_pair(engine, read_ext: str) -> bool:
    """
    Filter out invalid (engine, ext) pairs instead of skipping, as that
    produces 500+ pytest.skips.
    """
    engine = engine.values[0]
    if engine == 'openpyxl' and read_ext == '.xls':
        return False
    if engine == 'odf' and read_ext != '.ods':
        return False
    if read_ext == '.ods' and engine not in {'odf', 'calamine'}:
        return False
    if engine == 'pyxlsb' and read_ext != '.xlsb':
        return False
    if read_ext == '.xlsb' and engine not in {'pyxlsb', 'calamine'}:
        return False
    if engine == 'xlrd' and read_ext != '.xls':
        return False
    return True