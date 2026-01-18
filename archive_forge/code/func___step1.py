import sys
import copy
from typing import Union, NewType, Sequence, Tuple, Optional, Callable
def __step1(self) -> int:
    """
        For each row of the matrix, find the smallest element and
        subtract it from every element in its row. Go to Step 2.
        """
    C = self.C
    n = self.n
    for i in range(n):
        vals = [x for x in self.C[i] if x is not DISALLOWED]
        if len(vals) == 0:
            raise UnsolvableMatrix('Row {0} is entirely DISALLOWED.'.format(i))
        minval = min(vals)
        for j in range(n):
            if self.C[i][j] is not DISALLOWED:
                self.C[i][j] -= minval
    return 2