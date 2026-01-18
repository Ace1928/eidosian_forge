from rdkit import Chem
def SetProp(self, nm, val):
    Chem.Mol.SetProp(self, nm, str(val))