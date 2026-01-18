from itertools import combinations
from sympy.combinatorics.graycode import GrayCode
def _ranklex(self, subset_index, i, n):
    if subset_index == [] or i > n:
        return 0
    if i in subset_index:
        subset_index.remove(i)
        return 1 + _ranklex(self, subset_index, i + 1, n)
    return 2 ** (n - i - 1) + _ranklex(self, subset_index, i + 1, n)