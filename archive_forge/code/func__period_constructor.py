from datetime import (
import re
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas._libs.tslibs.parsing import DateParseError
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
@staticmethod
def _period_constructor(bound, offset):
    return Period(year=bound.year, month=bound.month, day=bound.day, hour=bound.hour, minute=bound.minute, second=bound.second + offset, freq='us')