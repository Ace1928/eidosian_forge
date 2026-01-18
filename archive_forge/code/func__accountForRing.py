from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.ChemUtils.DescriptorUtilities import setDescriptorVersion
from collections import defaultdict
def _accountForRing(self, atom):
    """Calculates the ring score for a single atom in a molecule"""
    if atom.GetIsAromatic():
        return 1
    if atom.IsInRing():
        return 2
    return 1