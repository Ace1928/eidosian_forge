import os
from collections.abc import Sequence
from numbers import Integral
import numpy as np
from .. import draw
from skimage import morphology
def _decompose_size(size, kernel_size=3):
    """Determine number of repeated iterations for a `kernel_size` kernel.

    Returns how many repeated morphology operations with an element of size
    `kernel_size` is equivalent to a morphology with a single kernel of size
    `n`.

    """
    if kernel_size % 2 != 1:
        raise ValueError('only odd length kernel_size is supported')
    return 1 + (size - kernel_size) // (kernel_size - 1)