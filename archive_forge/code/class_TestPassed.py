import pytest
import modin.pandas as pd
from modin.tests.pandas.utils import default_to_pandas_ignore_string
class TestPassed(BaseException):
    pass