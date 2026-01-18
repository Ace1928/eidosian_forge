from __future__ import annotations
from collections import abc
from datetime import (
from decimal import Decimal
import operator
import os
from typing import (
from dateutil.tz import (
import hypothesis
from hypothesis import strategies as st
import numpy as np
import pytest
from pytz import (
from pandas._config.config import _get_option
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.indexes.api import (
from pandas.util.version import Version
import zoneinfo
def _create_mi_with_dt64tz_level():
    """
    MultiIndex with a level that is a tzaware DatetimeIndex.
    """
    return MultiIndex.from_product([[1, 2], ['a', 'b'], date_range('20130101', periods=3, tz='US/Eastern')], names=['one', 'two', 'three'])