from collections import deque
from copy import copy
import numpy as np
from Bio.PDB.PDBExceptions import PDBConstructionException
def _generate_full_id(self):
    """Generate full_id (PRIVATE).

        Generate the full_id of the Entity based on its
        Id and the IDs of the parents.
        """
    entity_id = self.get_id()
    parts = [entity_id]
    parent = self.get_parent()
    while parent is not None:
        entity_id = parent.get_id()
        parts.append(entity_id)
        parent = parent.get_parent()
    parts.reverse()
    return tuple(parts)