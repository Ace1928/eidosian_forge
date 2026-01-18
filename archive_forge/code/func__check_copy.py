from string import ascii_lowercase
import copy
import pickle
import numpy as np
import pennylane as qml
from pennylane.operation import EigvalsUndefinedError
def _check_copy(op):
    """Check that copies and deep copies give identical objects."""
    copied_op = copy.copy(op)
    assert qml.equal(copied_op, op), 'copied op must be equal with qml.equal'
    assert copied_op == op, 'copied op must be equivalent to original operation'
    assert copied_op is not op, 'copied op must be a separate instance from original operaiton'
    assert qml.equal(copy.deepcopy(op), op), 'deep copied op must also be equal'