import datetime
import time
import re
import numbers
import functools
import contextlib
from numbers import Number
from typing import Union, Tuple, Iterable
from typing import cast
def _parse_timedelta_nanos(str):
    parts = re.finditer('(?P<value>[\\d.:]+)\\s?(?P<unit>[^\\W\\d_]+)?', str)
    chk_parts = _check_unmatched(parts, str)
    deltas = map(_parse_timedelta_part, chk_parts)
    return sum(deltas, _Saved_NS())