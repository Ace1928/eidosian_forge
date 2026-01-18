from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas._libs import lib
from pandas._libs.algos import unique_deltas
from pandas._libs.tslibs import (
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.dtypes import (
from pandas._libs.tslibs.fields import (
from pandas._libs.tslibs.offsets import (
from pandas._libs.tslibs.parsing import get_rule_month
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.common import is_numeric_dtype
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.algorithms import unique
def _get_wom_rule(self) -> str | None:
    weekdays = unique(self.index.weekday)
    if len(weekdays) > 1:
        return None
    week_of_months = unique((self.index.day - 1) // 7)
    week_of_months = week_of_months[week_of_months < 4]
    if len(week_of_months) == 0 or len(week_of_months) > 1:
        return None
    week = week_of_months[0] + 1
    wd = int_to_weekday[weekdays[0]]
    return f'WOM-{week}{wd}'