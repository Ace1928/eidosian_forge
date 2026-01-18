import enum
import hashlib
import json
import logging
import re
from typing import Iterable, Optional
from rdkit import Chem
from rdkit.Chem import rdMolHash
def _GetCanonicalAtomRanksAndBonds(mol, useSmilesOrdering=True):
    """
    returns a 2-tuple with:

    1. the canonical ranks of a molecule's atoms
    2. the bonds expressed as (canonical_atom_rank_1,canonical_atom_rank_2) where
       canonical_atom_rank_1 < canonical_atom_rank_2

    If useSmilesOrdering is True then the atom indices here correspond to the order of
    the atoms in the canonical SMILES, otherwise just the canonical atom order is used.
    useSmilesOrdering=True is a bit slower, but it allows the output to be linked to the
    canonical SMILES, which can be useful.

    """
    if mol.GetNumAtoms() == 0:
        return ([], [])
    if not useSmilesOrdering:
        atRanks = list(Chem.CanonicalRankAtoms(mol))
    else:
        smi = Chem.MolToSmiles(mol)
        ordertxt = mol.GetProp('_smilesAtomOutputOrder')
        smiOrder = [int(x) for x in ordertxt[1:-1].split(',') if x]
        atRanks = [0] * len(smiOrder)
        for i, idx in enumerate(smiOrder):
            atRanks[idx] = i
    bndOrder = []
    for bnd in mol.GetBonds():
        bo = atRanks[bnd.GetBeginAtomIdx()]
        eo = atRanks[bnd.GetEndAtomIdx()]
        if bo > eo:
            bo, eo = (eo, bo)
        bndOrder.append((bo, eo))
    return (atRanks, bndOrder)