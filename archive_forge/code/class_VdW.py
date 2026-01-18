import numpy as np
from numpy import linalg
from ase import units 
class VdW:

    def __init__(self, atomi, atomj, epsilonij=None, sigmaij=None, rminij=None, Aij=None, Bij=None, epsiloni=None, epsilonj=None, sigmai=None, sigmaj=None, rmini=None, rminj=None, scale=1.0):
        self.atomi = atomi
        self.atomj = atomj
        if epsilonij is not None:
            if sigmaij is not None:
                self.Aij = scale * 4.0 * epsilonij * sigmaij ** 12
                self.Bij = scale * 4.0 * epsilonij * sigmaij ** 6 * scale
            elif rminij is not None:
                self.Aij = scale * epsilonij * rminij ** 12
                self.Bij = scale * 2.0 * epsilonij * rminij ** 6
            else:
                raise NotImplementedError('not implemented combinationof vdW parameters.')
        elif Aij is not None and Bij is not None:
            self.Aij = scale * Aij
            self.Bij = scale * Bij
        elif epsiloni is not None and epsilonj is not None:
            if sigmai is not None and sigmaj is not None:
                self.Aij = scale * 4.0 * np.sqrt(epsiloni * epsilonj) * ((sigmai + sigmaj) / 2.0) ** 12
                self.Bij = scale * 2.0 * np.sqrt(epsiloni * epsilonj) * ((sigmai + sigmaj) / 2.0) ** 6
            elif rmini is not None and rminj is not None:
                self.Aij = scale * np.sqrt(epsiloni * epsilonj) * ((rmini + rminj) / 2.0) ** 12
                self.Bij = scale * 2.0 * np.sqrt(epsiloni * epsilonj) * ((rmini + rminj) / 2.0) ** 6
        else:
            raise NotImplementedError('not implemented combinationof vdW parameters.')
        self.r = None