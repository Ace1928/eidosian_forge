import random
from typing import Tuple, Optional
import numpy as np
import pytest
import cirq
def _get_assert_bidiagonalized_by_str(m, p, q, d):
    return f'm.round(3) : {np.round(m, 3)}, p.round(3) : {np.round(p, 3)}, q.round(3): {np.round(q, 3)}, np.abs(p.T @ m @ p).round(2): {np.abs(d).round(2)}'