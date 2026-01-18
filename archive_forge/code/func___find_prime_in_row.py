import sys
import copy
from typing import Union, NewType, Sequence, Tuple, Optional, Callable
def __find_prime_in_row(self, row) -> int:
    """
        Find the first prime element in the specified row. Returns
        the column index, or -1 if no starred element was found.
        """
    col = -1
    for j in range(self.n):
        if self.marked[row][j] == 2:
            col = j
            break
    return col