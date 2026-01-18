import numpy
from rdkit import Chem
def MaxEStateIndex(mol, force=1):
    return max(EStateIndices(mol, force))