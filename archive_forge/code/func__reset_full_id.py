from collections import deque
from copy import copy
import numpy as np
from Bio.PDB.PDBExceptions import PDBConstructionException
def _reset_full_id(self):
    """Reset the full_id (PRIVATE).

        Resets the full_id of this entity and
        recursively of all its children based on their ID.
        """
    for child in self:
        try:
            child._reset_full_id()
        except AttributeError:
            pass
    self.full_id = self._generate_full_id()