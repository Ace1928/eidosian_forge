import os.path
import re
from rdkit import Chem, RDConfig
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import (GetAtomPairFingerprint,
def AssignPattyTypes(mol, defns=None):
    """

    >>> from rdkit import Chem
    >>> AssignPattyTypes(Chem.MolFromSmiles('OCC(=O)O'))
    ['POL', 'HYD', 'OTH', 'ANI', 'ANI']

    """
    global _pattyDefs
    if defns is None:
        if _pattyDefs is None:
            _pattyDefs = _readPattyDefs()
        defns = _pattyDefs
    res = [''] * mol.GetNumAtoms()
    for matcher, nm in defns:
        matches = mol.GetSubstructMatches(matcher, uniquify=False)
        for match in matches:
            res[match[0]] = nm
    return res