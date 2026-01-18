import sys
import copy
from typing import Union, NewType, Sequence, Tuple, Optional, Callable
def __step2(self) -> int:
    """
        Find a zero (Z) in the resulting matrix. If there is no starred
        zero in its row or column, star Z. Repeat for each element in the
        matrix. Go to Step 3.
        """
    n = self.n
    for i in range(n):
        for j in range(n):
            if self.C[i][j] == 0 and (not self.col_covered[j]) and (not self.row_covered[i]):
                self.marked[i][j] = 1
                self.col_covered[j] = True
                self.row_covered[i] = True
                break
    self.__clear_covers()
    return 3