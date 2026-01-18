from __future__ import annotations
import os
import pytest
from pandas.compat._optional import VERSIONS
from pandas import (
import pandas._testing as tm
class PyArrowParser(BaseParser):
    engine = 'pyarrow'
    float_precision_choices = [None]