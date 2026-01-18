import itertools
from typing import Optional
import pytest
import numpy as np
import cirq
import cirq.testing
from cirq import linalg
class BasicStateVector(cirq.StateVectorMixin):

    def state_vector(self, copy: Optional[bool]=None) -> np.ndarray:
        return np.array([0, 1, 0, 0])