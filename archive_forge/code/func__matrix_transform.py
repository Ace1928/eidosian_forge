from typing import Sequence, Callable, Union
from functools import partial
from warnings import warn
import pennylane as qml
from pennylane.transforms.op_transforms import OperationTransformError
from pennylane import transform
from pennylane.typing import TensorLike
from pennylane.operation import Operator
from pennylane.pauli import PauliWord, PauliSentence
@partial(transform, is_informative=True)
def _matrix_transform(tape: qml.tape.QuantumTape, wire_order=None, **kwargs) -> (Sequence[qml.tape.QuantumTape], Callable):
    if not tape.wires:
        raise qml.operation.MatrixUndefinedError
    if wire_order and (not set(tape.wires).issubset(wire_order)):
        raise OperationTransformError(f'Wires in circuit {list(tape.wires)} are inconsistent with those in wire_order {list(wire_order)}')
    wires = kwargs.get('device_wires', None) or tape.wires
    wire_order = wire_order or wires

    def processing_fn(res):
        """Defines how matrix works if applied to a tape containing multiple operations."""
        params = res[0].get_parameters(trainable_only=False)
        interface = qml.math.get_interface(*params)
        if len(res[0].operations) == 0:
            result = qml.math.eye(2 ** len(wire_order), like=interface)
        else:
            result = matrix(res[0].operations[0], wire_order=wire_order)
        for op in res[0].operations[1:]:
            U = matrix(op, wire_order=wire_order)
            result = qml.math.matmul(*qml.math.coerce([U, result], like=interface), like=interface)
        return result
    return ([tape], processing_fn)