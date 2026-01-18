from typing import List, Union, Sequence
import warnings
import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.typing import TensorLike
from pennylane.ops import functions
from .parametrized_hamiltonian import ParametrizedHamiltonian
from .hardware_hamiltonian import HardwareHamiltonian
def _format_arraylike(x):
    for i, mat in enumerate(cache['matrices']):
        if qml.math.shape(x) == qml.math.shape(mat) and qml.math.allclose(x, mat):
            return f'M{i}'
    mat_num = len(cache['matrices'])
    cache['matrices'].append(x)
    return f'M{mat_num}'