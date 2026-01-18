import numpy as np
from ase.ga import get_raw_score
class AtomsComparator:
    """Compares the Atoms objects directly."""

    def looks_like(self, a1, a2):
        return a1 == a2