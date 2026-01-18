from dataclasses import replace
from functools import partial
from typing import Union, Tuple, Sequence
import concurrent.futures
import numpy as np
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.typing import Result, ResultBatch
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.transforms.core import TransformProgram
from pennylane.measurements import (
from pennylane.ops.qubit.observables import BasisStateProjector
from . import Device
from .execution_config import ExecutionConfig, DefaultExecutionConfig
from .default_qubit import accepted_sample_measurement
from .modifiers import single_tape_support, simulator_tracking
from .preprocess import (
@staticmethod
def _measure_stabilizer_entropy(stabilizer, wires, log_base=None):
    """Computes the Rényi entanglement entropy using stabilizer information.

        Computes the Rényi entanglement entropy :math:`S_A` for a subsytem described
        by :math:`A`, :math:`S_A = \\text{rank}(\\text{proj}_A {\\mathcal{S}}) - |A|`,
        where :math:`\\mathcal{S}` is the stabilizer group for the system using the theory
        described in Appendix A.1.d of `arXiv:1901.08092 <https://arxiv.org/abs/1901.08092>`_.

        Args:
            stabilizer (TensorLike): stabilizer set for the system
            wires (Iterable): wires describing the subsystem
            log_base (int): base for the logarithm.

        Returns:
            (float): entanglement entropy of the subsystem
        """
    num_qubits = qml.math.shape(stabilizer)[0]
    if len(wires) == num_qubits:
        return 0.0
    pauli_dict = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}
    terms = [qml.pauli.PauliWord({idx: pauli_dict[ele] for idx, ele in enumerate(row)}) for row in stabilizer]
    binary_mat = qml.pauli.utils._binary_matrix_from_pws(terms, num_qubits)
    partition_mat = qml.math.hstack((binary_mat[:, num_qubits:][:, wires], binary_mat[:, :num_qubits][:, wires]))
    rank = qml.math.sum(qml.math.any(qml.qchem.tapering._reduced_row_echelon(partition_mat), axis=1))
    entropy = qml.math.log(2) * (rank - len(wires))
    if log_base is None:
        return entropy
    return entropy / qml.math.log(log_base)