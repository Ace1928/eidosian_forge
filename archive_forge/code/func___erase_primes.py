import sys
import copy
from typing import Union, NewType, Sequence, Tuple, Optional, Callable
def __erase_primes(self) -> None:
    """Erase all prime markings"""
    for i in range(self.n):
        for j in range(self.n):
            if self.marked[i][j] == 2:
                self.marked[i][j] = 0