from pathlib import Path
from typing import List, Sequence
from warnings import warn
import numpy as np
from pennylane_lightning.core.lightning_base import (
class LightningQubit(LightningBaseFallBack):
    name = 'Lightning qubit PennyLane plugin [No binaries found - Fallback: default.qubit]'
    short_name = 'lightning.qubit'

    def __init__(self, wires, *, c_dtype=np.complex128, **kwargs):
        warn('Pre-compiled binaries for lightning.qubit are not available. Falling back to using the Python-based default.qubit implementation. To manually compile from source, follow the instructions at https://pennylane-lightning.readthedocs.io/en/latest/installation.html.', UserWarning)
        super().__init__(wires, c_dtype=c_dtype, **kwargs)