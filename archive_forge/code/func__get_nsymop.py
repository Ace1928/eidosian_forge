import os
import warnings
from functools import total_ordering
from typing import Union
import numpy as np
def _get_nsymop(self):
    """Returns total number of symmetry operations."""
    if self.centrosymmetric:
        return 2 * len(self._rotations) * len(self._subtrans)
    else:
        return len(self._rotations) * len(self._subtrans)