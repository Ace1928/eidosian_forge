from collections import defaultdict
from typing import Iterator
from .logic import Logic, And, Or, Not
def _tell(self, k, v):
    """Add fact k=v to the knowledge base.

        Returns True if the KB has actually been updated, False otherwise.
        """
    if k in self and self[k] is not None:
        if self[k] == v:
            return False
        else:
            raise InconsistentAssumptions(self, k, v)
    else:
        self[k] = v
        return True