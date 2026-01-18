from rdkit import Chem
def GetBond(self, idx):
    if self._onPatt is None:
        return None
    return self._onPatt.GetBondWithIdx(idx)