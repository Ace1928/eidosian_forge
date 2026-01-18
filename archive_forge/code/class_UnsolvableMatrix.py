import sys
import copy
from typing import Union, NewType, Sequence, Tuple, Optional, Callable
class UnsolvableMatrix(Exception):
    """
    Exception raised for unsolvable matrices
    """
    pass