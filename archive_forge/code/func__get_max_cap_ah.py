from __future__ import annotations
import math
from collections import defaultdict
import scipy.constants as const
from pymatgen.core import Composition, Element, Species
def _get_max_cap_ah(self, remove, insert):
    """Give max capacity in mAh for inserting and removing a charged ion
        This method does not normalize the capacity and intended as a helper method.
        """
    num_working_ions = 0
    if remove:
        num_working_ions += self.max_ion_removal
    if insert:
        num_working_ions += self.max_ion_insertion
    return num_working_ions * abs(self.working_ion_charge) * ELECTRON_TO_AMPERE_HOURS