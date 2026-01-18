from datetime import (
from functools import partial
from io import BytesIO
import os
import re
import numpy as np
import pytest
from pandas.compat import is_platform_windows
from pandas.compat._constants import PY310
from pandas.compat._optional import import_optional_dependency
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.excel import (
from pandas.io.excel._util import _writers
@td.skip_if_no('xlrd')
@td.skip_if_no('openpyxl')
class TestFSPath:

    def test_excelfile_fspath(self):
        with tm.ensure_clean('foo.xlsx') as path:
            df = DataFrame({'A': [1, 2]})
            df.to_excel(path)
            with ExcelFile(path) as xl:
                result = os.fspath(xl)
            assert result == path

    def test_excelwriter_fspath(self):
        with tm.ensure_clean('foo.xlsx') as path:
            with ExcelWriter(path) as writer:
                assert os.fspath(writer) == str(path)

    def test_to_excel_pos_args_deprecation(self):
        df = DataFrame({'a': [1, 2, 3]})
        msg = "Starting with pandas version 3.0 all arguments of to_excel except for the argument 'excel_writer' will be keyword-only."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            buf = BytesIO()
            writer = ExcelWriter(buf)
            df.to_excel(writer, 'Sheet_name_1')