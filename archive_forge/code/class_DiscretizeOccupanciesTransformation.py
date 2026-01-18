from __future__ import annotations
import logging
from fractions import Fraction
from typing import TYPE_CHECKING
import numpy as np
from numpy import around
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.elasticity.strain import Deformation
from pymatgen.analysis.ewald import EwaldMinimizer, EwaldSummation
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, get_el_sp
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.site_transformations import PartialRemoveSitesTransformation
from pymatgen.transformations.transformation_abc import AbstractTransformation
class DiscretizeOccupanciesTransformation(AbstractTransformation):
    """Discretizes the site occupancies in a disordered structure; useful for
    grouping similar structures or as a pre-processing step for order-disorder
    transformations.
    """

    def __init__(self, max_denominator=5, tol: float | None=None, fix_denominator=False):
        """
        Args:
            max_denominator:
                An integer maximum denominator for discretization. A higher
                denominator allows for finer resolution in the site occupancies.
            tol:
                A float that sets the maximum difference between the original and
                discretized occupancies before throwing an error. If None, it is
                set to 1 / (4 * max_denominator).
            fix_denominator(bool):
                If True, will enforce a common denominator for all species.
                This prevents a mix of denominators (for example, 1/3, 1/4)
                that might require large cell sizes to perform an enumeration.
                'tol' needs to be > 1.0 in some cases.
        """
        self.max_denominator = max_denominator
        self.tol = tol if tol is not None else 1 / (4 * max_denominator)
        self.fix_denominator = fix_denominator

    def apply_transformation(self, structure):
        """Discretizes the site occupancies in the structure.

        Args:
            structure: disordered Structure to discretize occupancies

        Returns:
            A new disordered Structure with occupancies discretized
        """
        if structure.is_ordered:
            return structure
        species = [dict(sp) for sp in structure.species_and_occu]
        for sp in species:
            for k in sp:
                old_occ = sp[k]
                new_occ = float(Fraction(old_occ).limit_denominator(self.max_denominator))
                if self.fix_denominator:
                    new_occ = around(old_occ * self.max_denominator) / self.max_denominator
                if round(abs(old_occ - new_occ), 6) > self.tol:
                    raise RuntimeError('Cannot discretize structure within tolerance!')
                sp[k] = new_occ
        return Structure(structure.lattice, species, structure.frac_coords)

    def __repr__(self):
        return 'DiscretizeOccupanciesTransformation'

    @property
    def inverse(self):
        """Returns: None."""
        return

    @property
    def is_one_to_many(self) -> bool:
        """Returns: False."""
        return False