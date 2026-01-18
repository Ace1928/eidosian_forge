import functools
import inspect
import os
import warnings
import pennylane as qml
class OperationTransformError(Exception):
    """Raised when there is an error with the op_transform logic"""