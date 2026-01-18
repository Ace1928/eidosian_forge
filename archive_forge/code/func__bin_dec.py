from typing import Dict, Optional, Sequence
import numpy as np
import cirq
from cirq import circuits
def _bin_dec(x: Optional[int], num_bits: int) -> str:
    if x is None:
        return 'None'
    return f'0b{bin(x)[2:].zfill(num_bits)} ({x})'