from __future__ import annotations
import typing
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
import numpy
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit import Instruction, Parameter, ParameterVector, ParameterExpression
from qiskit.exceptions import QiskitError
from ..blueprintcircuit import BlueprintCircuit
def _build_rotation_layer(self, circuit, param_iter, i):
    """Build a rotation layer."""
    if self._skip_unentangled_qubits:
        unentangled_qubits = self.get_unentangled_qubits()
    for j, block in enumerate(self.rotation_blocks):
        layer = QuantumCircuit(*self.qregs)
        block_indices = [list(range(k * block.num_qubits, (k + 1) * block.num_qubits)) for k in range(self.num_qubits // block.num_qubits)]
        if self._skip_unentangled_qubits:
            block_indices = [indices for indices in block_indices if set(indices).isdisjoint(unentangled_qubits)]
        for indices in block_indices:
            parameterized_block = self._parameterize_block(block, param_iter, i, j, indices)
            layer.compose(parameterized_block, indices, inplace=True)
        circuit.compose(layer, inplace=True)