import random
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule
def GetStereoisomerCount(m, options=StereoEnumerationOptions()):
    """ returns an estimate (upper bound) of the number of possible stereoisomers for a molecule

   Arguments:
      - m: the molecule to work with
      - options: parameters controlling the enumeration


    >>> from rdkit import Chem
    >>> from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
    >>> m = Chem.MolFromSmiles('BrC(Cl)(F)CCC(O)C')
    >>> GetStereoisomerCount(m)
    4
    >>> m = Chem.MolFromSmiles('CC(Cl)(O)C')
    >>> GetStereoisomerCount(m)
    1

    double bond stereochemistry is also included:

    >>> m = Chem.MolFromSmiles('BrC(Cl)(F)C=CC(O)C')
    >>> GetStereoisomerCount(m)
    8

    """
    tm = Chem.Mol(m)
    flippers = _getFlippers(tm, options)
    return 2 ** len(flippers)