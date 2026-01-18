from datetime import datetime
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries.holiday import (
class testCalendar(AbstractHolidayCalendar):
    rules = [USLaborDay]