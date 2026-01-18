import numpy as np
from autograd.extend import VSpace
from autograd.builtins import NamedTupleVSpace
class EigResultVSpace(NamedTupleVSpace):
    seq_type = np.linalg.linalg.EigResult