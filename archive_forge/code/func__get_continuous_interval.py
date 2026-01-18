import inspect
import itertools
import logging
import math
import sys
import weakref
from pyomo.common.pyomo_typing import overload
from pyomo.common.collections import ComponentSet
from pyomo.common.deprecation import deprecated, deprecation_warning, RenamedClass
from pyomo.common.errors import DeveloperError, PyomoException
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.sorting import sorted_robust
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import (
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.base.initializer import (
from pyomo.core.base.range import (
from pyomo.core.base.component import (
from pyomo.core.base.indexed_component import (
from pyomo.core.base.global_set import (
from collections.abc import Sequence
from operator import itemgetter
def _get_continuous_interval(self):
    ranges = []
    discrete = []
    for r in self.ranges():
        if r.isdiscrete():
            discrete.append(r)
        else:
            ranges.append(NumericRange(r.start, r.end, r.step, r.closed))
    if len(ranges) == 1 and (not discrete):
        r = ranges[0]
        return (None if r.start == -_inf else r.start, None if r.end == _inf else r.end, abs(r.step))
    for r in ranges:
        if not r.closed[0]:
            for d in discrete:
                if r.start in d:
                    r.closed = (True, r.closed[1])
                    break
        if not r.closed[1]:
            for d in discrete:
                if r.end in d:
                    r.closed = (r.closed[0], True)
                    break
    nRanges = len(ranges)
    r = ranges.pop()
    interval = NumericRange(r.start, r.end, r.step, r.closed)
    _rlen = len(ranges)
    while _rlen and nRanges > _rlen:
        nRanges = _rlen
        for i, r in enumerate(ranges):
            if interval.isdisjoint(r):
                continue
            ranges[i] = None
            if r.start < interval.start:
                interval.start = r.start
                interval.closed = (r.closed[0], interval.closed[1])
            elif not interval.closed[0] and r.start == interval.start:
                interval.closed = (r.closed[0], interval.closed[1])
            if r.end > interval.end:
                interval.end = r.end
                interval.closed = (interval.closed[0], r.closed[1])
            elif not interval.closed[1] and r.end == interval.end:
                interval.closed = (interval.closed[0], r.closed[1])
        ranges = list((_ for _ in ranges if _ is not None))
        _rlen = len(ranges)
    if ranges:
        return self.bounds() + (None,)
    for r in discrete:
        if not r.issubset(interval):
            return self.bounds() + (None,)
    start = interval.start
    if start == -_inf:
        start = None
    end = interval.end
    if end == _inf:
        end = None
    return (start, end, interval.step)