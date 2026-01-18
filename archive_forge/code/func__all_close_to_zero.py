import numpy as np
import autograd
import pennylane as qml
def _all_close_to_zero(dy):
    """
    Check if all entries of dy are close to 0. dy can also be a nested tuple
    structure of tensors, in which case this returns True iff all tensors are
    close to 0
    """
    if not isinstance(dy, (list, tuple)):
        return qml.math.allclose(dy, 0)
    return all((_all_close_to_zero(dy_) for dy_ in dy))