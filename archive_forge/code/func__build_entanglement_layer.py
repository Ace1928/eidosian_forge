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
def _build_entanglement_layer(self, circuit, param_iter, i):
    """Build an entanglement layer."""
    for j, block in enumerate(self.entanglement_blocks):
        layer = QuantumCircuit(*self.qregs)
        entangler_map = self.get_entangler_map(i, j, block.num_qubits)
        for indices in entangler_map:
            parameterized_block = self._parameterize_block(block, param_iter, i, j, indices)
            layer.compose(parameterized_block, indices, inplace=True)
        circuit.compose(layer, inplace=True)