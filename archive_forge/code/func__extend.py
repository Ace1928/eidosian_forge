from CuSignal under terms of the MIT license.
import warnings
from typing import Set
import cupy
import numpy as np
def _extend(M, sym):
    """Extend window by 1 sample if needed for DFT-even symmetry"""
    if not sym:
        return (M + 1, True)
    else:
        return (M, False)