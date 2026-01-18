from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
def HeavyAtomCount(mol):
    """ Number of heavy atoms a molecule."""
    return mol.GetNumHeavyAtoms()