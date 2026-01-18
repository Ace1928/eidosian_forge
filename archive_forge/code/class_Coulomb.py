import numpy as np
from numpy import linalg
from ase import units 
class Coulomb:

    def __init__(self, atomi, atomj, chargeij=None, chargei=None, chargej=None, scale=1.0):
        self.atomi = atomi
        self.atomj = atomj
        if chargeij is not None:
            self.chargeij = scale * chargeij * 8987551787.368176 * units.m * units.J / units.C / units.C
        elif chargei is not None and chargej is not None:
            self.chargeij = scale * chargei * chargej * 8987551787.368176 * units.m * units.J / units.C / units.C
        else:
            raise NotImplementedError('not implemented combinationof Coulomb parameters.')
        self.r = None