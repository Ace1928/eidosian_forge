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
def _parameterize_block(self, block, param_iter=None, rep_num=None, block_num=None, indices=None, params=None):
    """Convert ``block`` to a circuit of correct width and parameterized using the iterator."""
    if self._overwrite_block_parameters:
        if params is None:
            params = self._parameter_generator(rep_num, block_num, indices)
        if params is None:
            params = [next(param_iter) for _ in range(len(get_parameters(block)))]
        update = dict(zip(block.parameters, params))
        return block.assign_parameters(update)
    return block.copy()