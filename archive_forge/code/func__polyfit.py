from copy import copy
from typing import Any, Dict, Optional, Sequence, Callable
from pennylane import apply, adjoint
from pennylane.math import mean, shape, round
from pennylane.queuing import AnnotatedQueue
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.transforms import transform
import pennylane as qml
def _polyfit(x, y, order):
    """Brute force implementation of polynomial fit"""
    x = qml.math.convert_like(x, y[0])
    x = qml.math.cast_like(x, y[0])
    X = qml.math.vander(x, order + 1)
    y = qml.math.stack(y)
    scale = qml.math.sum(qml.math.sqrt(X * X), axis=0)
    X = X / scale
    c = qml.math.linalg.pinv(qml.math.transpose(X) @ X)
    c = c @ qml.math.transpose(X)
    c = qml.math.tensordot(c, y, axes=1)
    c = qml.math.transpose(qml.math.transpose(c) / scale)
    return c