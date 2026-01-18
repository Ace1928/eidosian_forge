from typing import Sequence, Callable
import functools
from functools import partial
import warnings
import numpy as np
import pennylane as qml
from pennylane.circuit_graph import LayerData
from pennylane.queuing import WrappedObj
from pennylane.transforms import transform
@functools.lru_cache()
def _get_gen_op(op, allow_nonunitary, aux_wire):
    """Get the controlled-generator operation for a given operation.

    Args:
        op (WrappedObj[Operation]): Wrapped Operation from which to extract the generator. The
            Operation needs to be wrapped for hashability in order to use the lru-cache.
        allow_nonunitary (bool): Whether non-unitary gates are allowed in the circuit
        aux_wire (int or pennylane.wires.Wires): Auxiliary wire on which to control the operation

    Returns
        qml.Operation: Controlled-generator operation of the generator of ``op``, controlled
        on wire ``aux_wire``.

    Raises
        ValueError: If the generator of ``op`` is not known or it is non-unitary while
        ``allow_nonunitary=False``.

    If ``allow_nonunitary=True``, a general :class:`~.pennylane.ControlledQubitUnitary` is returned,
    otherwise only controlled Pauli operations are used. If the operation has a non-unitary
    generator but ``allow_nonunitary=False``, the operation ``op`` should have been decomposed
    before, leading to a ``ValueError``.
    """
    op_to_cgen = {qml.RX: qml.CNOT, qml.RY: qml.CY, qml.RZ: qml.CZ, qml.PhaseShift: qml.CZ}
    op = op.obj
    try:
        cgen = op_to_cgen[op.__class__]
        return cgen(wires=[aux_wire, *op.wires])
    except KeyError as e:
        if allow_nonunitary:
            mat = qml.matrix(qml.generator(op)[0])
            return qml.ControlledQubitUnitary(mat, control_wires=aux_wire, wires=op.wires)
        raise ValueError(f'Generator for operation {op} not known and non-unitary operations deactivated via allow_nonunitary=False.') from e