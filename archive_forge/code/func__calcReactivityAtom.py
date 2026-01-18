import itertools
from collections import Counter, defaultdict
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdqueries
from . import utils
def _calcReactivityAtom(self, a):
    if self._isSp3OrAromaticCarbon(a) or (len(a.GetNeighbors()) == 0 and a.GetFormalCharge() == 0):
        return 0
    reactivity = 1
    b = a.GetBonds()
    if self._isHeteroAtom(a) or a.GetTotalNumHs() > 0:
        reactivity += 1
    if a.IsInRing():
        if a.GetIsAromatic():
            reactivity += 0.5
    else:
        reactivity += 1
    if a.GetFormalCharge():
        reactivity += 2
    for bo in b:
        ni = bo.GetOtherAtom(a)
        if bo.GetBondTypeAsDouble() > 1.5:
            reactivity += 1
            if ni.GetTotalNumHs() > 0:
                reactivity += 1
        if self._isHeteroAtom(ni):
            reactivity += 1
            if a.GetAtomicNum() in (7, 8) and ni.GetAtomicNum() in (7, 8):
                reactivity += 2
            elif ni.GetAtomicNum() in (12, 14, 15, 46, 50):
                reactivity += 1
    return reactivity