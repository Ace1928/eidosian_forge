import enum
import hashlib
import json
import logging
import re
from typing import Iterable, Optional
from rdkit import Chem
from rdkit.Chem import rdMolHash
def _CanonicalizeCOPSGroup(sg, atRanks, sortAtomAndBondOrder):
    """
    NOTES: if sortAtomAndBondOrder is true then the atom and bond lists will be sorted.
    This assumes that the ordering of those lists is not important
    """
    if sg.GetProp('TYPE') != 'COP':
        return None
    ats = tuple((atRanks[x] for x in sg.GetAtoms()))
    if sortAtomAndBondOrder:
        ats = tuple(sorted(ats))
    props = sg.GetPropsAsDict()
    res = dict(type='COP', atoms=ats, index=props.get('index', 0))
    return res