from string import ascii_lowercase
import copy
import pickle
import numpy as np
import pennylane as qml
from pennylane.operation import EigvalsUndefinedError
def _check_bind_new_parameters(op):
    """Check that bind new parameters can create a new op with different data."""
    new_data = [d * 0.0 for d in op.data]
    new_data_op = qml.ops.functions.bind_new_parameters(op, new_data)
    failure_comment = 'bind_new_parameters must be able to update the operator with new data.'
    for d1, d2 in zip(new_data_op.data, new_data):
        assert qml.math.allclose(d1, d2), failure_comment