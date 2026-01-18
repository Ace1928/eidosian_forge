from __future__ import annotations
import copy
import itertools
import logging
import math
import warnings
from collections import defaultdict
from collections.abc import Sequence
from fractions import Fraction
from functools import lru_cache
from math import cos, sin
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
import scipy.cluster
import spglib
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Molecule, PeriodicSite, Structure
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.util.coord import find_in_coord_list, pbc_diff
from pymatgen.util.due import Doi, due
class SpacegroupOperations(list):
    """Represents a space group, which is a collection of symmetry operations."""

    def __init__(self, int_symbol, int_number, symmops):
        """
        Args:
            int_symbol (str): International symbol of the spacegroup.
            int_number (int): International number of the spacegroup.
            symmops ([SymmOp]): Symmetry operations associated with the
                spacegroup.
        """
        self.int_symbol = int_symbol
        self.int_number = int_number
        super().__init__(symmops)

    def are_symmetrically_equivalent(self, sites1, sites2, symm_prec=0.001) -> bool:
        """Given two sets of PeriodicSites, test if they are actually symmetrically
        equivalent under this space group. Useful, for example, if you want to test if
        selecting atoms 1 and 2 out of a set of 4 atoms are symmetrically the same as
        selecting atoms 3 and 4, etc.

        One use is in PartialRemoveSpecie transformation to return only
        symmetrically distinct arrangements of atoms.

        Args:
            sites1 ([PeriodicSite]): 1st set of sites
            sites2 ([PeriodicSite]): 2nd set of sites
            symm_prec (float): Tolerance in atomic distance to test if atoms
                are symmetrically similar.

        Returns:
            bool: Whether the two sets of sites are symmetrically equivalent.
        """

        def in_sites(site):
            return any((test_site.is_periodic_image(site, symm_prec, check_lattice=False) for test_site in sites1))
        for op in self:
            new_sites2 = [PeriodicSite(site.species, op.operate(site.frac_coords), site.lattice) for site in sites2]
            for site in new_sites2:
                if not in_sites(site):
                    break
            else:
                return True
        return False

    def __str__(self):
        return f'{self.int_symbol} ({self.int_number}) spacegroup'