from pathlib import Path
from typing import List, Sequence
from warnings import warn
import numpy as np
from pennylane_lightning.core.lightning_base import (
def _state_dtype(dtype):
    if dtype not in [np.complex128, np.complex64]:
        raise ValueError(f'Data type is not supported for state-vector computation: {dtype}')
    return StateVectorC128 if dtype == np.complex128 else StateVectorC64