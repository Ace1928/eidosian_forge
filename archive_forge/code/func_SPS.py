from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.ChemUtils.DescriptorUtilities import setDescriptorVersion
from collections import defaultdict
@setDescriptorVersion(version='1.0.0')
def SPS(mol, normalize=True):
    """Calculates the SpacialScore descriptor. By default, the score is normalized by the number of heavy atoms (nSPS) resulting in a float value,
    otherwise (normalize=False) the absolute score is returned as an integer.
    """
    return _SpacialScore(mol, normalize=normalize).score