from __future__ import annotations
import abc
import itertools
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.json import MSONable
from pymatgen.core import Composition, Lattice, Structure, get_el_sp
from pymatgen.optimization.linear_assignment import LinearAssignment
from pymatgen.util.coord import lattice_points_in_supercell
from pymatgen.util.coord_cython import is_coord_subset_pbc, pbc_shortest_vectors
class ElementComparator(AbstractComparator):
    """
    A Comparator that matches elements. i.e. oxidation states are
    ignored.
    """

    def are_equal(self, sp1, sp2) -> bool:
        """
        True if element:amounts are exactly the same, i.e.,
        oxidation state is not considered.

        Args:
            sp1: First species. A dict of {specie/element: amt} as per the
                definition in Site and PeriodicSite.
            sp2: Second species. A dict of {specie/element: amt} as per the
                definition in Site and PeriodicSite.

        Returns:
            Boolean indicating whether species are the same based on element
            and amounts.
        """
        comp1 = Composition(sp1)
        comp2 = Composition(sp2)
        return comp1.get_el_amt_dict() == comp2.get_el_amt_dict()

    def get_hash(self, composition):
        """Returns: Fractional element composition."""
        return composition.element_composition.fractional_composition