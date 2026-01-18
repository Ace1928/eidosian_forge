from collections import \
import rdkit.Chem.ChemUtils.DescriptorUtilities as _du
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdMolDescriptors as _rdMolDescriptors
from rdkit.Chem import rdPartialCharges
from rdkit.Chem.EState.EState import (MaxAbsEStateIndex, MaxEStateIndex,
from rdkit.Chem.QED import qed
from rdkit.Chem.SpacialScore import SPS
def NumValenceElectrons(mol):
    """ The number of valence electrons the molecule has

    >>> NumValenceElectrons(Chem.MolFromSmiles('CC'))
    14
    >>> NumValenceElectrons(Chem.MolFromSmiles('C(=O)O'))
    18
    >>> NumValenceElectrons(Chem.MolFromSmiles('C(=O)[O-]'))
    18
    >>> NumValenceElectrons(Chem.MolFromSmiles('C(=O)'))
    12

    """
    tbl = Chem.GetPeriodicTable()
    return sum((tbl.GetNOuterElecs(atom.GetAtomicNum()) - atom.GetFormalCharge() + atom.GetTotalNumHs() for atom in mol.GetAtoms()))