import sys
import copy
from typing import Union, NewType, Sequence, Tuple, Optional, Callable
def __find_star_in_col(self, col: Sequence[AnyNum]) -> int:
    """
        Find the first starred element in the specified row. Returns
        the row index, or -1 if no starred element was found.
        """
    row = -1
    for i in range(self.n):
        if self.marked[i][col] == 1:
            row = i
            break
    return row