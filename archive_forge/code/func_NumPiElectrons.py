import math
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
def NumPiElectrons(atom):
    """ Returns the number of electrons an atom is using for pi bonding

    >>> m = Chem.MolFromSmiles('C=C')
    >>> NumPiElectrons(m.GetAtomWithIdx(0))
    1

    >>> m = Chem.MolFromSmiles('C#CC')
    >>> NumPiElectrons(m.GetAtomWithIdx(0))
    2
    >>> NumPiElectrons(m.GetAtomWithIdx(1))
    2

    >>> m = Chem.MolFromSmiles('O=C=CC')
    >>> NumPiElectrons(m.GetAtomWithIdx(0))
    1
    >>> NumPiElectrons(m.GetAtomWithIdx(1))
    2
    >>> NumPiElectrons(m.GetAtomWithIdx(2))
    1
    >>> NumPiElectrons(m.GetAtomWithIdx(3))
    0

    >>> m = Chem.MolFromSmiles('c1ccccc1')
    >>> NumPiElectrons(m.GetAtomWithIdx(0))
    1

    FIX: this behaves oddly in these cases:

    >>> m = Chem.MolFromSmiles('S(=O)(=O)')
    >>> NumPiElectrons(m.GetAtomWithIdx(0))
    2

    >>> m = Chem.MolFromSmiles('S(=O)(=O)(O)O')
    >>> NumPiElectrons(m.GetAtomWithIdx(0))
    0

    In the second case, the S atom is tagged as sp3 hybridized.

    """
    res = 0
    if atom.GetIsAromatic():
        res = 1
    elif atom.GetHybridization() != Chem.HybridizationType.SP3:
        res = atom.GetExplicitValence() - atom.GetNumExplicitHs()
        if res < atom.GetDegree():
            raise ValueError('explicit valence exceeds atom degree')
        res -= atom.GetDegree()
    return res