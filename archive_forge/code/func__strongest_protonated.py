from warnings import warn
import copy
import logging
from rdkit import Chem
from .utils import memoized_property
def _strongest_protonated(self, mol):
    for position, pair in enumerate(self.acid_base_pairs):
        for occurrence in mol.GetSubstructMatches(pair.acid):
            return (position, occurrence)
    return (None, None)