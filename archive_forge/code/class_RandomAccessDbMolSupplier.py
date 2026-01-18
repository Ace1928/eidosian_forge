import sys
from rdkit import Chem
from rdkit.Chem.Suppliers.MolSupplier import MolSupplier
class RandomAccessDbMolSupplier(DbMolSupplier):

    def __init__(self, dbResults, **kwargs):
        """

      DbResults should be a Dbase.DbResultSet.RandomAccessDbResultSet

    """
        DbMolSupplier.__init__(self, dbResults, **kwargs)
        self._pos = -1

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        newD = self._data[idx]
        return self._BuildMol(newD)

    def Reset(self):
        self._pos = -1

    def NextMol(self):
        self._pos += 1
        res = None
        if self._pos < len(self):
            res = self[self._pos]
        return res