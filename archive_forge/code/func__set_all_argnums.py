from functools import partial
from typing import Callable, List, Tuple, Optional, Sequence, Union
import pennylane as qml
from pennylane.typing import Result, ResultBatch
from pennylane.tape import QuantumTape
from .transform_dispatcher import TransformContainer, TransformError, TransformDispatcher
def _set_all_argnums(self, qnode, args, kwargs, argnums):
    """It can be used inside the QNode to set all argnums (tape level) using argnums from the argnums at the QNode
        level.
        """
    argnums_list = []
    for index, transform in enumerate(self):
        argnums = [0] if qnode.interface in ['jax', 'jax-jit'] and argnums is None else argnums
        if (transform._use_argnum or transform.classical_cotransform) and argnums:
            params = qml.math.jax_argnums_to_tape_trainable(qnode, argnums, TransformProgram(self[0:index]), args, kwargs)
            argnums_list.append([qml.math.get_trainable_indices(param) for param in params])
        else:
            argnums_list.append(None)
    self._argnums = argnums_list
    qnode.construct(args, kwargs)