from functools import wraps
from inspect import signature
import warnings
import numpy as np
from autoray import numpy as anp
import pennylane as qml
def _reconstruction(x):
    """Univariate reconstruction based on arbitrary shifts."""
    x = x - x0
    return a0 + qml.math.tensordot(qml.math.cos(spectrum * x), a, axes=[[0], [0]]) + qml.math.tensordot(qml.math.sin(spectrum * x), b, axes=[[0], [0]])