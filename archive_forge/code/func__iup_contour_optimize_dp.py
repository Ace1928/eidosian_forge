from typing import (
from numbers import Integral, Real
@cython.locals(i=cython.int, j=cython.int, best_cost=cython.double, best_j=cython.int, cost=cython.double, forced=set, tolerance=cython.double)
def _iup_contour_optimize_dp(deltas: _DeltaSegment, coords: _PointSegment, forced=set(), tolerance: Real=0, lookback: Integral=None):
    """Straightforward Dynamic-Programming.  For each index i, find least-costly encoding of
    points 0 to i where i is explicitly encoded.  We find this by considering all previous
    explicit points j and check whether interpolation can fill points between j and i.

    Note that solution always encodes last point explicitly.  Higher-level is responsible
    for removing that restriction.

    As major speedup, we stop looking further whenever we see a "forced" point."""
    n = len(deltas)
    if lookback is None:
        lookback = n
    lookback = min(lookback, MAX_LOOKBACK)
    costs = {-1: 0}
    chain = {-1: None}
    for i in range(0, n):
        best_cost = costs[i - 1] + 1
        costs[i] = best_cost
        chain[i] = i - 1
        if i - 1 in forced:
            continue
        for j in range(i - 2, max(i - lookback, -2), -1):
            cost = costs[j] + 1
            if cost < best_cost and can_iup_in_between(deltas, coords, j, i, tolerance):
                costs[i] = best_cost = cost
                chain[i] = j
            if j in forced:
                break
    return (chain, costs)