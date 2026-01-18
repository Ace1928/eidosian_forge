from dataclasses import dataclass
from fractions import Fraction
from numbers import Complex
from typing import (
import numpy as np
def _cis(x: ExpressionValueDesignator) -> complex:
    return cast(complex, np.exp(1j * x))