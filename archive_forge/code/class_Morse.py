import numpy as np
from numpy import linalg
from ase import units 
class Morse:

    def __init__(self, atomi, atomj, D, alpha, r0):
        self.atomi = atomi
        self.atomj = atomj
        self.D = D
        self.alpha = alpha
        self.r0 = r0
        self.r = None