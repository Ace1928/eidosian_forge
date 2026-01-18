import numpy as np
import pytest
import cirq
class HasChannel:

    def _kraus_(self):
        return (np.sqrt(0.5) * np.eye(4, dtype=np.complex128), np.sqrt(0.5) * u)