from warnings import warn
import copy
import logging
from rdkit import Chem
from .utils import memoized_property
def _weakest_ionized(self, mol):
    for position, pair in enumerate(reversed(self.acid_base_pairs)):
        for occurrence in mol.GetSubstructMatches(pair.base):
            return (len(self.acid_base_pairs) - position - 1, occurrence)
    return (None, None)