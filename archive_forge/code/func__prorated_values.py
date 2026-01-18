import datetime
import time
import re
import numbers
import functools
import contextlib
from numbers import Number
from typing import Union, Tuple, Iterable
from typing import cast
def _prorated_values(rate: str) -> Iterable[Tuple[str, Number]]:
    """
    Given a rate (a string in units per unit time), and return that same
    rate for various time periods.

    >>> for period, value in _prorated_values('20/hour'):
    ...     print('{period}: {value:0.3f}'.format(**locals()))
    minute: 0.333
    hour: 20.000
    day: 480.000
    month: 14609.694
    year: 175316.333

    """
    match = re.match('(?P<value>[\\d.]+)/(?P<period>\\w+)$', rate)
    res = cast(re.Match, match).groupdict()
    value = float(res['value'])
    value_per_second = value / get_period_seconds(res['period'])
    for period in ('minute', 'hour', 'day', 'month', 'year'):
        period_value = value_per_second * get_period_seconds(period)
        yield (period, period_value)