import math
import numpy
from rdkit import Chem
from rdkit.Chem import Graphs, rdchem, rdMolDescriptors
from rdkit.ML.InfoTheory import entropy
def _hkDeltas(mol, skipHs=1):
    global ptable
    res = []
    if hasattr(mol, '_hkDeltas') and mol._hkDeltas is not None:
        return mol._hkDeltas
    for atom in mol.GetAtoms():
        n = atom.GetAtomicNum()
        if n > 1:
            nV = ptable.GetNOuterElecs(n)
            nHs = atom.GetTotalNumHs()
            if n <= 10:
                res.append(float(nV - nHs))
            else:
                res.append(float(nV - nHs) / float(n - nV - 1))
        elif n == 1:
            if not skipHs:
                res.append(0.0)
        else:
            res.append(0.0)
    mol._hkDeltas = res
    return res