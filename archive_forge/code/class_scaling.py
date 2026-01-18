from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
class scaling(object):
    """
    Matrices for diagonal scaling

    Attributes
    ----------
    D        - matrix in R^{n \\times n}
    E        - matrix in R^{m \\times n}
    Dinv     - inverse of D
    Einv     - inverse of E
    c        - cost scaling
    cinv    - inverse of cost scaling
    """

    def __init__(self):
        self.D = None
        self.E = None
        self.Dinv = None
        self.Einv = None
        self.c = None
        self.cinv = None