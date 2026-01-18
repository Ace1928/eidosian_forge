import pickle
from rdkit import DataStructs
from rdkit.VLib.Node import VLibNode
def NextItem(self):
    self._pos += 1
    res = None
    if self._pos < len(self):
        res = self[self._pos]
    return res