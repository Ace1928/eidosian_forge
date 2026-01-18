import numpy as np
import pytest
import cirq
class HasQuditUnitary:

    def _qid_shape_(self):
        return (3,)

    def _unitary_(self) -> np.ndarray:
        raise NotImplementedError