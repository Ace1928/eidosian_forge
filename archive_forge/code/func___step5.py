import sys
import copy
from typing import Union, NewType, Sequence, Tuple, Optional, Callable
def __step5(self) -> int:
    """
        Construct a series of alternating primed and starred zeros as
        follows. Let Z0 represent the uncovered primed zero found in Step 4.
        Let Z1 denote the starred zero in the column of Z0 (if any).
        Let Z2 denote the primed zero in the row of Z1 (there will always
        be one). Continue until the series terminates at a primed zero
        that has no starred zero in its column. Unstar each starred zero
        of the series, star each primed zero of the series, erase all
        primes and uncover every line in the matrix. Return to Step 3
        """
    count = 0
    path = self.path
    path[count][0] = self.Z0_r
    path[count][1] = self.Z0_c
    done = False
    while not done:
        row = self.__find_star_in_col(path[count][1])
        if row >= 0:
            count += 1
            path[count][0] = row
            path[count][1] = path[count - 1][1]
        else:
            done = True
        if not done:
            col = self.__find_prime_in_row(path[count][0])
            count += 1
            path[count][0] = path[count - 1][0]
            path[count][1] = col
    self.__convert_path(path, count)
    self.__clear_covers()
    self.__erase_primes()
    return 3