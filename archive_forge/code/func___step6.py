import sys
import copy
from typing import Union, NewType, Sequence, Tuple, Optional, Callable
def __step6(self) -> int:
    """
        Add the value found in Step 4 to every element of each covered
        row, and subtract it from every element of each uncovered column.
        Return to Step 4 without altering any stars, primes, or covered
        lines.
        """
    minval = self.__find_smallest()
    events = 0
    for i in range(self.n):
        for j in range(self.n):
            if self.C[i][j] is DISALLOWED:
                continue
            if self.row_covered[i]:
                self.C[i][j] += minval
                events += 1
            if not self.col_covered[j]:
                self.C[i][j] -= minval
                events += 1
            if self.row_covered[i] and (not self.col_covered[j]):
                events -= 2
    if events == 0:
        raise UnsolvableMatrix('Matrix cannot be solved!')
    return 4