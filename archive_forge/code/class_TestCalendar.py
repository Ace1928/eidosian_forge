from datetime import datetime
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries.holiday import (
class TestCalendar(AbstractHolidayCalendar):

    def __init__(self, name=None, rules=None) -> None:
        super().__init__(name=name, rules=rules)