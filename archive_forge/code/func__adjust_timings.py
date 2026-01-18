import re
import operator
import heapq
from collections import namedtuple
from collections.abc import Sequence
from contextlib import contextmanager
from functools import cached_property
from numba.core import config
import llvmlite.binding as llvm
def _adjust_timings(records):
    """Adjust timing records because of truncated information.

    Details: The percent information can be used to improve the timing
    information.

    Returns
    -------
    res: List[PassTimingRecord]
    """
    total_rec = records[-1]
    assert total_rec.pass_name == 'Total'

    def make_adjuster(attr):
        time_attr = f'{attr}_time'
        percent_attr = f'{attr}_percent'
        time_getter = operator.attrgetter(time_attr)

        def adjust(d):
            """Compute percent x total_time = adjusted"""
            total = time_getter(total_rec)
            adjusted = total * d[percent_attr] * 0.01
            d[time_attr] = adjusted
            return d
        return adjust
    adj_fns = [make_adjuster(x) for x in ['user', 'system', 'user_system', 'wall']]
    dicts = map(lambda x: x._asdict(), records)

    def chained(d):
        for fn in adj_fns:
            d = fn(d)
        return PassTimingRecord(**d)
    return list(map(chained, dicts))