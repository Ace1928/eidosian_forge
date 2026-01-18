from typing import Sequence, Callable
from functools import partial
import numpy as np
import pennylane as qml
from pennylane.gradients.metric_tensor import _get_aux_wire
from pennylane import transform
from pennylane.gradients.gradient_transform import _contract_qjac_with_cjac
from pennylane.transforms.tape_expand import expand_invalid_trainable_hadamard_gradient
from .gradient_transform import (
def _expand_transform_hadamard(tape: qml.tape.QuantumTape, argnum=None, aux_wire=None, device_wires=None) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Expand function to be applied before hadamard gradient."""
    expanded_tape = expand_invalid_trainable_hadamard_gradient(tape)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]
    return ([expanded_tape], null_postprocessing)