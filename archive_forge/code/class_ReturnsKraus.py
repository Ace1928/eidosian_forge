from typing import Iterable, List, Sequence, Tuple
import numpy as np
import pytest
import cirq
class ReturnsKraus:

    def _kraus_(self) -> Sequence[np.ndarray]:
        return c