from __future__ import annotations
import collections
import itertools
from math import acos, pi
from typing import TYPE_CHECKING
from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi
from pymatgen.analysis.local_env import JmolNN, VoronoiNN
from pymatgen.core import Composition, Element, PeriodicSite, Species
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class RelaxationAnalyzer:
    """This class analyzes the relaxation in a calculation."""

    def __init__(self, initial_structure: Structure, final_structure: Structure) -> None:
        """Please note that the input and final structures should have the same
        ordering of sites. This is typically the case for most computational codes.

        Args:
            initial_structure (Structure): Initial input structure to
                calculation.
            final_structure (Structure): Final output structure from
                calculation.

        Raises:
            ValueError: If initial and final structures have different formulas.
        """
        if final_structure.formula != initial_structure.formula:
            raise ValueError('Initial and final structures have different formulas!')
        self.initial = initial_structure
        self.final = final_structure

    def get_percentage_volume_change(self) -> float:
        """
        Returns the percentage volume change.

        Returns:
            float: Volume change in percent. 0.055 means a 5.5% increase.
        """
        return self.final.volume / self.initial.volume - 1

    def get_percentage_lattice_parameter_changes(self) -> dict[str, float]:
        """
        Returns the percentage lattice parameter changes.

        Returns:
            dict[str, float]: Percent changes in lattice parameter, e.g.,
                {'a': 0.012, 'b': 0.021, 'c': -0.031} implies a change of 1.2%,
                2.1% and -3.1% in the a, b and c lattice parameters respectively.
        """
        initial_latt = self.initial.lattice
        final_latt = self.final.lattice
        return {length: getattr(final_latt, length) / getattr(initial_latt, length) - 1 for length in ['a', 'b', 'c']}

    def get_percentage_bond_dist_changes(self, max_radius: float=3.0) -> dict[int, dict[int, float]]:
        """
        Returns the percentage bond distance changes for each site up to a
        maximum radius for nearest neighbors.

        Args:
            max_radius (float): Maximum radius to search for nearest
               neighbors. This radius is applied to the initial structure,
               not the final structure.

        Returns:
            dict[int, dict[int, float]]: Bond distance changes in the form {index1: {index2: 0.011, ...}}.
                For economy of representation, the index1 is always less than index2, i.e., since bonding
                between site1 and site_n is the same as bonding between site_n and site1, there is no
                reason to duplicate the information or computation.
        """
        data: dict[int, dict[int, float]] = collections.defaultdict(dict)
        for indices in itertools.combinations(list(range(len(self.initial))), 2):
            ii, jj = sorted(indices)
            initial_dist = self.initial[ii].distance(self.initial[jj])
            if initial_dist < max_radius:
                final_dist = self.final[ii].distance(self.final[jj])
                data[ii][jj] = final_dist / initial_dist - 1
        return data