from functools import partial
from typing import Callable, List, Tuple, Optional, Sequence, Union
import pennylane as qml
from pennylane.typing import Result, ResultBatch
from pennylane.tape import QuantumTape
from .transform_dispatcher import TransformContainer, TransformError, TransformDispatcher
def _jacobian(*args, **kwargs):
    return jax.jacobian(classical_function, argnums=argnums)(*args, **kwargs)