from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.AtomPairs import Utils
from rdkit.Chem.rdMolDescriptors import (
def GetTopologicalTorsionFingerprintAsIds(mol, targetSize=4):
    nonZeroElements = GetTopologicalTorsionFingerprint(mol, targetSize).GetNonzeroElements()
    frequencies = sorted(nonZeroElements.items())
    res = []
    for k, v in frequencies:
        res.extend([k] * v)
    return res