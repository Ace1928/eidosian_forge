import collections
import numpy as np
from numba.core import types
@wrap
def _run_quicksort(A):
    stack = [Partition(zero, zero)] * 100
    stack[0] = Partition(zero, len(A) - 1)
    n = 1
    while n > 0:
        n -= 1
        low, high = stack[n]
        while high - low >= SMALL_QUICKSORT:
            assert n < MAX_STACK
            l, r = partition3(A, low, high)
            if r == high:
                high = l - 1
            elif l == low:
                low = r + 1
            elif high - r > l - low:
                stack[n] = Partition(r + 1, high)
                n += 1
                high = l - 1
            else:
                stack[n] = Partition(low, l - 1)
                n += 1
                low = r + 1
        insertion_sort(A, low, high)