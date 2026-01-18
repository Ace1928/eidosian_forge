import numpy as np
from autograd.extend import VSpace
from autograd.builtins import NamedTupleVSpace
def _inner_prod(self, x, y):
    return np.real(np.dot(np.conj(np.ravel(x)), np.ravel(y)))