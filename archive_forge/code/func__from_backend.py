from typing import Optional, List, Tuple, Iterable, Callable, Union, Dict
import numpy as np
from qiskit.exceptions import QiskitError
from ..distributions.quasi import QuasiDistribution
from ..counts import Counts
from .base_readout_mitigator import BaseReadoutMitigator
from .utils import counts_probability_vector, z_diagonal, str2diag
def _from_backend(self, backend, qubits):
    """Calculates amats from backend properties readout_error"""
    backend_qubits = backend.properties().qubits
    if qubits is not None:
        if any((qubit >= len(backend_qubits) for qubit in qubits)):
            raise QiskitError('The chosen backend does not contain the specified qubits.')
        reduced_backend_qubits = [backend_qubits[i] for i in qubits]
        backend_qubits = reduced_backend_qubits
    num_qubits = len(backend_qubits)
    amats = np.zeros([num_qubits, 2, 2], dtype=float)
    for qubit_idx, qubit_prop in enumerate(backend_qubits):
        for prop in qubit_prop:
            if prop.name == 'prob_meas0_prep1':
                amats[qubit_idx][0, 1] = prop.value
                amats[qubit_idx][1, 1] = 1 - prop.value
            if prop.name == 'prob_meas1_prep0':
                amats[qubit_idx][1, 0] = prop.value
                amats[qubit_idx][0, 0] = 1 - prop.value
    return amats