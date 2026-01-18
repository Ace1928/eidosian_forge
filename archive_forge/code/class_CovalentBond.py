from __future__ import annotations
import collections
import json
import os
import warnings
from typing import TYPE_CHECKING
from pymatgen.core import Element
class CovalentBond:
    """Defines a covalent bond between two sites."""

    def __init__(self, site1: Site, site2: Site) -> None:
        """Initializes a covalent bond between two sites.

        Args:
            site1 (Site): First site.
            site2 (Site): Second site.
        """
        self.site1 = site1
        self.site2 = site2

    @property
    def length(self) -> float:
        """Length of the bond."""
        return self.site1.distance(self.site2)

    def get_bond_order(self, tol: float=0.2, default_bl: float | None=None) -> float:
        """The bond order according the distance between the two sites.

        Args:
            tol (float): Relative tolerance to test.
                (1 + tol) * the longest bond distance is considered
                to be the threshold length for a bond to exist.
                (1 - tol) * the shortest bond distance is considered
                to be the shortest possible bond length
                Defaults to 0.2.
            default_bl: If a particular type of bond does not exist,
                use this bond length as a default value
                (bond order = 1). If None, a ValueError will be thrown.

        Returns:
            Float value of bond order. For example, for C-C bond in
            benzene, return 1.7.
        """
        sp1 = next(iter(self.site1.species))
        sp2 = next(iter(self.site2.species))
        dist = self.site1.distance(self.site2)
        return get_bond_order(sp1, sp2, dist, tol, default_bl)

    @staticmethod
    def is_bonded(site1, site2, tol: float=0.2, bond_order: float | None=None, default_bl: float | None=None):
        """Test if two sites are bonded, up to a certain limit.

        Args:
            site1 (Site): First site
            site2 (Site): Second site
            tol (float): Relative tolerance to test. Basically, the code
                checks if the distance between the sites is less than (1 +
                tol) * typical bond distances. Defaults to 0.2, i.e.,
                20% longer.
            bond_order: Bond order to test. If None, the code simply checks
                against all possible bond data. Defaults to None.
            default_bl: If a particular type of bond does not exist, use this
                bond length. If None, a ValueError will be thrown.

        Returns:
            Boolean indicating whether two sites are bonded.
        """
        sp1 = next(iter(site1.species))
        sp2 = next(iter(site2.species))
        dist = site1.distance(site2)
        syms = tuple(sorted([sp1.symbol, sp2.symbol]))
        if syms in bond_lengths:
            all_lengths = bond_lengths[syms]
            if bond_order:
                return dist < (1 + tol) * all_lengths[bond_order]
            return any((dist < (1 + tol) * v for v in all_lengths.values()))
        if default_bl:
            return dist < (1 + tol) * default_bl
        raise ValueError(f'No bond data for elements {syms[0]} - {syms[1]}')

    def __repr__(self) -> str:
        return f'Covalent bond between {self.site1} and {self.site2}'