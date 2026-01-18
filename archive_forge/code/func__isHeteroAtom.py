import itertools
from collections import Counter, defaultdict
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdqueries
from . import utils
def _isHeteroAtom(self, a):
    return a.GetAtomicNum() not in (6, 1)