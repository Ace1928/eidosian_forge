import numpy as np
from ase.ga import get_raw_score
class NoComparator:
    """Returns False always. If you don't want any comparator."""

    def looks_like(self, *args):
        return False