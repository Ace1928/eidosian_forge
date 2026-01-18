import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def _find_reentrances(self, reentrances):
    """
        Return a dictionary that maps from the ``id`` of each feature
        structure contained in ``self`` (including ``self``) to a
        boolean value, indicating whether it is reentrant or not.
        """
    if id(self) in reentrances:
        reentrances[id(self)] = True
    else:
        reentrances[id(self)] = False
        for fval in self._values():
            if isinstance(fval, FeatStruct):
                fval._find_reentrances(reentrances)
    return reentrances