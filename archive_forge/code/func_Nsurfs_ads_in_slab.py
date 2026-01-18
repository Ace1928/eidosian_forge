from __future__ import annotations
import copy
import itertools
import random
import warnings
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from sympy import Symbol
from sympy.solvers import linsolve, solve
from pymatgen.analysis.wulff import WulffShape
from pymatgen.core import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.surface import get_slab_regions
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp.outputs import Locpot, Outcar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
@property
def Nsurfs_ads_in_slab(self):
    """Returns the TOTAL number of adsorbed surfaces in the slab."""
    struct = self.structure
    weights = [s.species.weight for s in struct]
    center_of_mass = np.average(struct.frac_coords, weights=weights, axis=0)
    n_surfs = 0
    if any((site.species_string in self.ads_entries_dict for site in struct if site.frac_coords[2] > center_of_mass[2])):
        n_surfs += 1
    if any((site.species_string in self.ads_entries_dict for site in struct if site.frac_coords[2] < center_of_mass[2])):
        n_surfs += 1
    return n_surfs