from dataclasses import dataclass
from fractions import Fraction
from numbers import Complex
from typing import (
import numpy as np
def _complex_str(iq: Any) -> str:
    """Convert a number to a string."""
    if isinstance(iq, Complex):
        return f'{iq.real}' if iq.imag == 0.0 else f'{iq.real} + ({iq.imag})*i'
    else:
        return str(iq)