from rdkit import DataStructs, RDConfig, rdBase
from rdkit.Chem import rdchem
from rdkit.Geometry import rdGeometry
from rdkit.Chem.inchi import *
from rdkit.Chem.rdchem import *
from rdkit.Chem.rdCIPLabeler import *
from rdkit.Chem.rdmolfiles import *
from rdkit.Chem.rdmolops import *
class _GetRDKitObjIterator:

    def _sizeCalc(self):
        raise NotImplementedError()

    def _getRDKitItem(self, i):
        raise NotImplementedError()

    def __init__(self, mol):
        self._mol = mol
        self._pos = 0
        self._size = self._sizeCalc()

    def __len__(self):
        if self._sizeCalc() != self._size:
            raise RuntimeError('size changed during iteration')
        return self._size

    def __getitem__(self, i):
        if i < 0 or i >= len(self):
            raise IndexError('index out of range')
        return self._getRDKitItem(i)

    def __next__(self):
        if self._pos >= len(self):
            raise StopIteration
        ret = self[self._pos]
        self._pos += 1
        return ret

    def __iter__(self):
        for i in range(0, len(self)):
            self._pos = i
            yield self[i]
        self._pos = self._size