from string import ascii_lowercase
import copy
import pickle
import numpy as np
import pennylane as qml
from pennylane.operation import EigvalsUndefinedError
def _check_pickle(op):
    """Check that an operation can be dumped and reloaded with pickle."""
    pickled = pickle.dumps(op)
    unpickled = pickle.loads(pickled)
    assert unpickled == op, 'operation must be able to be pickled and unpickled'