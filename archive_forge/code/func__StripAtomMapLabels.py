import enum
import hashlib
import json
import logging
import re
from typing import Iterable, Optional
from rdkit import Chem
from rdkit.Chem import rdMolHash
def _StripAtomMapLabels(mol):
    for at in mol.GetAtoms():
        at.ClearProp(ATOM_PROP_MAP_NUMBER)