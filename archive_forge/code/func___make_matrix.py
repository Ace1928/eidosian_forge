import sys
import copy
from typing import Union, NewType, Sequence, Tuple, Optional, Callable
def __make_matrix(self, n: int, val: AnyNum) -> Matrix:
    """Create an *n*x*n* matrix, populating it with the specific value."""
    matrix = []
    for i in range(n):
        matrix += [[val for j in range(n)]]
    return matrix