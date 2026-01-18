from __future__ import annotations
import logging
import warnings
from fractions import Fraction
from functools import reduce
from itertools import chain, combinations, product
from math import cos, floor, gcd
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.fractions import lcm
from numpy.testing import assert_allclose
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core.lattice import Lattice
from pymatgen.core.sites import PeriodicSite, Site
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def _update_c(self, new_c: float) -> None:
    """Modifies the c-direction of the lattice without changing the site Cartesian coordinates
        Be careful you can mess up the interface by setting a c-length that can't accommodate all the sites.
        """
    if new_c <= 0:
        raise ValueError('New c-length must be greater than 0')
    new_latt_matrix = [*self.lattice.matrix[:2].tolist(), [0, 0, new_c]]
    new_lattice = Lattice(new_latt_matrix)
    self._lattice = new_lattice
    for site, c_coords in zip(self, self.cart_coords):
        site._lattice = new_lattice
        site.coords = c_coords