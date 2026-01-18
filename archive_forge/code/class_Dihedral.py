import numpy as np
from numpy import linalg
from ase import units 
class Dihedral:

    def __init__(self, atomi, atomj, atomk, atoml, k, d0=None, n=None, alpha=None, rref=None):
        self.atomi = atomi
        self.atomj = atomj
        self.atomk = atomk
        self.atoml = atoml
        self.k = k
        self.d0 = d0
        self.n = n
        self.alpha = alpha
        self.rref = rref
        self.d = None