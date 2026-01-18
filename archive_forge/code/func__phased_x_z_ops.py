from typing import Iterator, List, Optional
import itertools
import math
import numpy as np
import cirq
from cirq_google import ops
def _phased_x_z_ops(mat: np.ndarray, q: cirq.Qid) -> Iterator[cirq.Operation]:
    """Yields `cirq.PhasedXZGate` operation implementing `mat` if it is not identity."""
    gate = cirq.single_qubit_matrix_to_phxz(mat)
    if gate:
        yield gate(q)