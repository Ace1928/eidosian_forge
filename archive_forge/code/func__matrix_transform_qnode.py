from typing import Sequence, Callable, Union
from functools import partial
from warnings import warn
import pennylane as qml
from pennylane.transforms.op_transforms import OperationTransformError
from pennylane import transform
from pennylane.typing import TensorLike
from pennylane.operation import Operator
from pennylane.pauli import PauliWord, PauliSentence
@_matrix_transform.custom_qnode_transform
def _matrix_transform_qnode(self, qnode, targs, tkwargs):
    tkwargs.setdefault('device_wires', qnode.device.wires)
    return self.default_qnode_transform(qnode, targs, tkwargs)