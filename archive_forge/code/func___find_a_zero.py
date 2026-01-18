import sys
import copy
from typing import Union, NewType, Sequence, Tuple, Optional, Callable
def __find_a_zero(self, i0: int=0, j0: int=0) -> Tuple[int, int]:
    """Find the first uncovered element with value 0"""
    row = -1
    col = -1
    i = i0
    n = self.n
    done = False
    while not done:
        j = j0
        while True:
            if self.C[i][j] == 0 and (not self.row_covered[i]) and (not self.col_covered[j]):
                row = i
                col = j
                done = True
            j = (j + 1) % n
            if j == j0:
                break
        i = (i + 1) % n
        if i == i0:
            done = True
    return (row, col)