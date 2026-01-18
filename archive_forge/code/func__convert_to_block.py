from __future__ import annotations
import typing
from collections.abc import Callable, Sequence
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import Gate, Instruction, Parameter
from .n_local import NLocal
from ..standard_gates import (
def _convert_to_block(self, layer: str | type | Gate | QuantumCircuit) -> QuantumCircuit:
    """For a layer provided as str (e.g. ``'ry'``) or type (e.g. :class:`.RYGate`) this function
         returns the
         according layer type along with the number of parameters (e.g. ``(RYGate, 1)``).

        Args:
            layer: The qubit layer.

        Returns:
            The specified layer with the required number of parameters.

        Raises:
            TypeError: The type of ``layer`` is invalid.
            ValueError: The type of ``layer`` is str but the name is unknown.
            ValueError: The type of ``layer`` is type but the layer type is unknown.

        Note:
            Outlook: If layers knew their number of parameters as static property, we could also
            allow custom layer types.
        """
    if isinstance(layer, QuantumCircuit):
        return layer
    theta = Parameter('θ')
    valid_layers = {'ch': CHGate(), 'cx': CXGate(), 'cy': CYGate(), 'cz': CZGate(), 'crx': CRXGate(theta), 'cry': CRYGate(theta), 'crz': CRZGate(theta), 'h': HGate(), 'i': IGate(), 'id': IGate(), 'iden': IGate(), 'rx': RXGate(theta), 'rxx': RXXGate(theta), 'ry': RYGate(theta), 'ryy': RYYGate(theta), 'rz': RZGate(theta), 'rzx': RZXGate(theta), 'rzz': RZZGate(theta), 's': SGate(), 'sdg': SdgGate(), 'swap': SwapGate(), 'x': XGate(), 'y': YGate(), 'z': ZGate(), 't': TGate(), 'tdg': TdgGate()}
    if isinstance(layer, str):
        try:
            layer = valid_layers[layer]
        except KeyError as ex:
            raise ValueError(f'Unknown layer name `{layer}`.') from ex
    if isinstance(layer, type):
        instance = None
        for gate in valid_layers.values():
            if isinstance(gate, layer):
                instance = gate
        if instance is None:
            raise ValueError(f'Unknown layer type`{layer}`.')
        layer = instance
    if isinstance(layer, Instruction):
        circuit = QuantumCircuit(layer.num_qubits)
        circuit.append(layer, list(range(layer.num_qubits)))
        return circuit
    raise TypeError(f'Invalid input type {type(layer)}. ' + '`layer` must be a type, str or QuantumCircuit.')