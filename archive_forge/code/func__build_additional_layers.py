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
def _build_additional_layers(self, circuit, which):
    if which == 'appended':
        blocks = self._appended_blocks
        entanglements = self._appended_entanglement
    elif which == 'prepended':
        blocks = reversed(self._prepended_blocks)
        entanglements = reversed(self._prepended_entanglement)
    else:
        raise ValueError('`which` must be either `appended` or `prepended`.')
    for block, ent in zip(blocks, entanglements):
        layer = QuantumCircuit(*self.qregs)
        if isinstance(ent, str):
            ent = get_entangler_map(block.num_qubits, self.num_qubits, ent)
        for indices in ent:
            layer.compose(block, indices, inplace=True)
        circuit.compose(layer, inplace=True)