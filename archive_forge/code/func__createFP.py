import itertools
from collections import Counter, defaultdict
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdqueries
from . import utils
def _createFP(self, mol, invariant, bitinfo, useBondTypes=True, radius=1):
    return AllChem.GetMorganFingerprint(mol=mol, radius=radius, invariants=invariant, useBondTypes=useBondTypes, bitInfo=bitinfo)