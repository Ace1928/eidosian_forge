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
def _get_quarterly_rule(self) -> str | None:
    if len(self.mdiffs) > 1:
        return None
    if not self.mdiffs[0] % 3 == 0:
        return None
    pos_check = self.month_position_check()
    if pos_check is None:
        return None
    else:
        return {'cs': 'QS', 'bs': 'BQS', 'ce': 'QE', 'be': 'BQE'}.get(pos_check)