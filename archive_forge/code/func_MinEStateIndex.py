import numpy
from rdkit import Chem
def MinEStateIndex(mol, force=1):
    return min(EStateIndices(mol, force))