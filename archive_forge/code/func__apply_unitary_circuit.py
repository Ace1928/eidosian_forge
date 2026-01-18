import abc
import enum
import html
import itertools
import math
from collections import defaultdict
from typing import (
from typing_extensions import Self
import networkx
import numpy as np
import cirq._version
from cirq import _compat, devices, ops, protocols, qis
from cirq._doc import document
from cirq.circuits._bucket_priority_queue import BucketPriorityQueue
from cirq.circuits.circuit_operation import CircuitOperation
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.circuits.qasm_output import QasmOutput
from cirq.circuits.text_diagram_drawer import TextDiagramDrawer
from cirq.circuits.moment import Moment
from cirq.protocols import circuit_diagram_info_protocol
from cirq.type_workarounds import NotImplementedType
def _apply_unitary_circuit(circuit: 'cirq.AbstractCircuit', state: np.ndarray, qubits: Tuple['cirq.Qid', ...], dtype: Type[np.complexfloating]) -> np.ndarray:
    """Applies a circuit's unitary effect to the given vector or matrix.

    This method assumes that the caller wants to ignore measurements.

    Args:
        circuit: The circuit to simulate. All operations must have a known
            matrix or decompositions leading to known matrices. Measurements
            are allowed to be in the circuit, but they will be ignored.
        state: The initial state tensor (i.e. superposition or unitary matrix).
            This is what will be left-multiplied by the circuit's effective
            unitary. If this is a state vector, it must have shape
            (2,) * num_qubits. If it is a unitary matrix it should have shape
            (2,) * (2*num_qubits).
        qubits: The qubits in the state tensor. Determines which axes operations
            apply to. An operation targeting the k'th qubit in this list will
            operate on the k'th axis of the state tensor.
        dtype: The numpy dtype to use for applying the unitary. Must be a
            complex dtype.

    Returns:
        The left-multiplied state tensor.
    """
    buffer = np.empty_like(state)

    def on_stuck(bad_op):
        return TypeError(f'Operation without a known matrix or decomposition: {bad_op!r}')
    unitary_ops = protocols.decompose(circuit.all_operations(), keep=protocols.has_unitary, intercepting_decomposer=_decompose_measurement_inversions, on_stuck_raise=on_stuck)
    result = protocols.apply_unitaries(unitary_ops, qubits, protocols.ApplyUnitaryArgs(state, buffer, range(len(qubits))))
    assert result is not None, 'apply_unitaries() should raise TypeError instead'
    return result