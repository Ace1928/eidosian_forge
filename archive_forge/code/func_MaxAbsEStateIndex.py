import numpy
from rdkit import Chem
def MaxAbsEStateIndex(mol, force=1):
    return max((abs(x) for x in EStateIndices(mol, force)))