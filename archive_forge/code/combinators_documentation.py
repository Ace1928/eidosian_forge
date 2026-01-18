import functools
from typing import Union, TYPE_CHECKING
import numpy as np
from cirq._doc import document
Concatenates blocks into a block diagonal matrix.

    Args:
        *blocks: Square matrices to place along the diagonal of the result.

    Returns:
        A block diagonal matrix with the given blocks along its diagonal.

    Raises:
        ValueError: A block isn't square.
    