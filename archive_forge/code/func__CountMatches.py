import os
from rdkit import Chem, RDConfig
def _CountMatches(mol, patt, unique=True):
    return len(mol.GetSubstructMatches(patt, uniquify=unique))