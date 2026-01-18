from collections import \
import rdkit.Chem.ChemUtils.DescriptorUtilities as _du
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdMolDescriptors as _rdMolDescriptors
from rdkit.Chem import rdPartialCharges
from rdkit.Chem.EState.EState import (MaxAbsEStateIndex, MaxEStateIndex,
from rdkit.Chem.QED import qed
from rdkit.Chem.SpacialScore import SPS
def _isCallable(thing):
    return isinstance(thing, abc.Callable) or hasattr(thing, '__call__')