import copy
import os
import pickle
import warnings
import numpy as np
def _interpretIndexes(self, ind):
    if not isinstance(ind, tuple):
        if isinstance(ind, list) and len(ind) > 0 and isinstance(ind[0], slice):
            ind = tuple(ind)
        else:
            ind = (ind,)
    nInd = [slice(None)] * self.ndim
    numOk = True
    for i in range(0, len(ind)):
        axis, index, isNamed = self._interpretIndex(ind[i], i, numOk)
        nInd[axis] = index
        if isNamed:
            numOk = False
    return tuple(nInd)