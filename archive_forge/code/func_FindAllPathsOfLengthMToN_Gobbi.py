import time
import unittest
import numpy
from scipy.optimize import linear_sum_assignment
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem.Fingerprints import FingerprintMols
def FindAllPathsOfLengthMToN_Gobbi(mol, minlength, maxlength, rootedAtAtom=-1, uniquepaths=True):
    """this function returns the same set of bond paths as the Gobbi paper.  These differ a little from the rdkit FindAllPathsOfLengthMToN function"""
    paths = []
    for atom in mol.GetAtoms():
        if rootedAtAtom == -1 or atom.GetIdx() == rootedAtAtom:
            path = []
            visited = set([atom.GetIdx()])
            _FindAllPathsOfLengthMToN_Gobbi(atom, path, minlength, maxlength, visited, paths)
    if uniquepaths:
        uniquepathlist = []
        seen = set()
        for path in paths:
            if path not in seen:
                reversepath = tuple([i for i in path[::-1]])
                if reversepath not in seen:
                    uniquepathlist.append(path)
                    seen.add(path)
        return uniquepathlist
    else:
        return paths