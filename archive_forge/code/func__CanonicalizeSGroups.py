import enum
import hashlib
import json
import logging
import re
from typing import Iterable, Optional
from rdkit import Chem
from rdkit.Chem import rdMolHash
def _CanonicalizeSGroups(mol, dataFieldNames=None, sortAtomAndBondOrder=True):
    """
    NOTES: if sortAtomAndBondOrder is true then the atom and bond lists will be sorted.
    This assumes that the ordering of those lists is not important
    """
    dataFieldNames = dataFieldNames or ['Atrop']
    atRanks, bndOrder = _GetCanonicalAtomRanksAndBonds(mol)
    res = []
    for sg in Chem.GetMolSubstanceGroups(mol):
        lres = None
        if sg.GetProp('TYPE') == 'DAT':
            lres = _CanonicalizeDataSGroup(sg, atRanks, bndOrder, dataFieldNames, sortAtomAndBondOrder)
        elif sg.GetProp('TYPE') == 'SRU':
            lres = _CanonicalizeSRUSGroup(mol, sg, atRanks, bndOrder, sortAtomAndBondOrder)
        elif sg.GetProp('TYPE') == 'COP':
            lres = _CanonicalizeCOPSGroup(sg, atRanks, sortAtomAndBondOrder)
        if lres is not None:
            res.append(lres)
    if len(res) > 1:
        tres = sorted((tuple(x.items()) for x in res))
        res = tuple((dict(x) for x in tres))
        idxmap = {}
        for i, itm in enumerate(res):
            if 'index' in itm:
                idxmap[itm['index']] = i + 1
                itm['index'] = i + 1
        for i, itm in enumerate(res):
            if 'PARENT' in itm:
                itm['PARENT'] = idxmap[itm['PARENT']]
    return json.dumps(res)