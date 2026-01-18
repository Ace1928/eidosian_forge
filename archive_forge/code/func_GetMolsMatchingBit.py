import sys
from rdkit import Chem
from rdkit.Chem.rdfragcatalog import *
def GetMolsMatchingBit(mols, bit, fps):
    res = []
    if isinstance(bit, BitGainsInfo):
        bitId = bit.id
    else:
        bitId = bit
    for i, mol in enumerate(mols):
        fp = fps[i]
        if fp[bitId]:
            res.append(mol)
    return res