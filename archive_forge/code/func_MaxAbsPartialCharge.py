from collections import \
import rdkit.Chem.ChemUtils.DescriptorUtilities as _du
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdMolDescriptors as _rdMolDescriptors
from rdkit.Chem import rdPartialCharges
from rdkit.Chem.EState.EState import (MaxAbsEStateIndex, MaxEStateIndex,
from rdkit.Chem.QED import qed
from rdkit.Chem.SpacialScore import SPS
def MaxAbsPartialCharge(mol, force=False):
    v1, v2 = _ChargeDescriptors(mol, force)
    return max(abs(v1), abs(v2))