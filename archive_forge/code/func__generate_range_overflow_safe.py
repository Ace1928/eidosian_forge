from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas._libs.lib import i8max
from pandas._libs.tslibs import (
def _generate_range_overflow_safe(endpoint: int, periods: int, stride: int, side: str='start') -> int:
    """
    Calculate the second endpoint for passing to np.arange, checking
    to avoid an integer overflow.  Catch OverflowError and re-raise
    as OutOfBoundsDatetime.

    Parameters
    ----------
    endpoint : int
        nanosecond timestamp of the known endpoint of the desired range
    periods : int
        number of periods in the desired range
    stride : int
        nanoseconds between periods in the desired range
    side : {'start', 'end'}
        which end of the range `endpoint` refers to

    Returns
    -------
    other_end : int

    Raises
    ------
    OutOfBoundsDatetime
    """
    assert side in ['start', 'end']
    i64max = np.uint64(i8max)
    msg = f'Cannot generate range with {side}={endpoint} and periods={periods}'
    with np.errstate(over='raise'):
        try:
            addend = np.uint64(periods) * np.uint64(np.abs(stride))
        except FloatingPointError as err:
            raise OutOfBoundsDatetime(msg) from err
    if np.abs(addend) <= i64max:
        return _generate_range_overflow_safe_signed(endpoint, periods, stride, side)
    elif endpoint > 0 and side == 'start' and (stride > 0) or (endpoint < 0 < stride and side == 'end'):
        raise OutOfBoundsDatetime(msg)
    elif side == 'end' and endpoint - stride <= i64max < endpoint:
        return _generate_range_overflow_safe(endpoint - stride, periods - 1, stride, side)
    mid_periods = periods // 2
    remaining = periods - mid_periods
    assert 0 < remaining < periods, (remaining, periods, endpoint, stride)
    midpoint = int(_generate_range_overflow_safe(endpoint, mid_periods, stride, side))
    return _generate_range_overflow_safe(midpoint, remaining, stride, side)