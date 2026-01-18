from functools import reduce
from nltk.parse import load_parser
from nltk.sem.logic import (
from nltk.sem.skolemize import skolemize
def _violates_constraints(self, label, ancestors):
    """
        Return True if the `label' cannot be placed underneath the holes given
        by the set `ancestors' because it would violate the constraints imposed
        on it.
        """
    for c in self.constraints:
        if c.lhs == label:
            if c.rhs not in ancestors:
                return True
    return False