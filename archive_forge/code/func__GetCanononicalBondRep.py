import enum
import hashlib
import json
import logging
import re
from typing import Iterable, Optional
from rdkit import Chem
from rdkit.Chem import rdMolHash
def _GetCanononicalBondRep(bond, atomRanks):
    aid1 = bond.GetBeginAtomIdx()
    aid2 = bond.GetEndAtomIdx()
    if atomRanks[aid1] > atomRanks[aid2] or (atomRanks[aid1] == atomRanks[aid2] and aid1 > aid2):
        aid1, aid2 = (aid2, aid1)
    return (aid1, aid2)