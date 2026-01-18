import sys
import copy
from typing import Union, NewType, Sequence, Tuple, Optional, Callable
def __find_star_in_row(self, row: Sequence[AnyNum]) -> int:
    """
        Find the first starred element in the specified row. Returns
        the column index, or -1 if no starred element was found.
        """
    col = -1
    for j in range(self.n):
        if self.marked[row][j] == 1:
            col = j
            break
    return col