import os
import sys
import tempfile
from rdkit import Chem
def GetAtomCoords(self, sels):
    """ returns the coordinates of the selected atoms """
    res = {}
    for label, idx in sels:
        coords = self.server.getAtomCoords('(%s and id %d)' % (label, idx))
        res[label, idx] = coords
    return res