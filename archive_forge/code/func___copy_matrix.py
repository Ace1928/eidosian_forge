import sys
import copy
from typing import Union, NewType, Sequence, Tuple, Optional, Callable
def __copy_matrix(self, matrix: Matrix) -> Matrix:
    """Return an exact copy of the supplied matrix"""
    return copy.deepcopy(matrix)