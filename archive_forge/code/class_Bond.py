import numpy as np
from numpy import linalg
from ase import units 
class Bond:

    def __init__(self, atomi, atomj, k, b0, alpha=None, rref=None):
        self.atomi = atomi
        self.atomj = atomj
        self.k = k
        self.b0 = b0
        self.alpha = alpha
        self.rref = rref
        self.b = None