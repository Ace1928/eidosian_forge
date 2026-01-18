from typing import Sequence, Callable
from itertools import chain
from functools import partial
import numpy as np
import pennylane as qml
from pennylane.gradients.metric_tensor import _contract_metric_tensor_with_cjac
from pennylane.transforms import transform
def _reshape_real_imag(state, dim):
    state = qml.math.reshape(state, (dim,))
    return (qml.math.real(state), qml.math.imag(state))