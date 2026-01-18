import sys
from rdkit import Chem
def TypeAtoms(mol):
    """  assigns each atom in a molecule to an EState type

  **Returns:**

     list of tuples (atoms can possibly match multiple patterns) with atom types

  """
    if esPatterns is None:
        BuildPatts()
    nAtoms = mol.GetNumAtoms()
    res = [None] * nAtoms
    for name, patt in esPatterns:
        matches = mol.GetSubstructMatches(patt, uniquify=0)
        for match in matches:
            idx = match[0]
            if res[idx] is None:
                res[idx] = [name]
            elif name not in res[idx]:
                res[idx].append(name)
    for i, v in enumerate(res):
        if v is not None:
            res[i] = tuple(v)
        else:
            res[i] = tuple()
    return res