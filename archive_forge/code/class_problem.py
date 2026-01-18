from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
class problem(object):
    """
    QP problem of the form
        minimize	1/2 x' P x + q' x
        subject to	l <= A x <= u

    Attributes
    ----------
    P, q
    A, l, u
    """

    def __init__(self, dims, Pdata, Pindices, Pindptr, q, Adata, Aindices, Aindptr, l, u):
        self.n, self.m = dims
        self.P = spspa.csc_matrix((Pdata, Pindices, Pindptr), shape=(self.n, self.n))
        self.q = q
        self.A = spspa.csc_matrix((Adata, Aindices, Aindptr), shape=(self.m, self.n))
        self.l = l if l is not None else -np.inf * np.ones(self.m)
        self.u = u if u is not None else np.inf * np.ones(self.m)