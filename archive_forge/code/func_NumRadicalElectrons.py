from collections import \
import rdkit.Chem.ChemUtils.DescriptorUtilities as _du
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdMolDescriptors as _rdMolDescriptors
from rdkit.Chem import rdPartialCharges
from rdkit.Chem.EState.EState import (MaxAbsEStateIndex, MaxEStateIndex,
from rdkit.Chem.QED import qed
from rdkit.Chem.SpacialScore import SPS
def NumRadicalElectrons(mol):
    """ The number of radical electrons the molecule has
      (says nothing about spin state)

    >>> NumRadicalElectrons(Chem.MolFromSmiles('CC'))
    0
    >>> NumRadicalElectrons(Chem.MolFromSmiles('C[CH3]'))
    0
    >>> NumRadicalElectrons(Chem.MolFromSmiles('C[CH2]'))
    1
    >>> NumRadicalElectrons(Chem.MolFromSmiles('C[CH]'))
    2
    >>> NumRadicalElectrons(Chem.MolFromSmiles('C[C]'))
    3

    """
    return sum((atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()))