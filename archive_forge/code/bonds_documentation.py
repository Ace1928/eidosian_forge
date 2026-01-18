from __future__ import annotations
import collections
import json
import os
import warnings
from typing import TYPE_CHECKING
from pymatgen.core import Element
Test if two sites are bonded, up to a certain limit.

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
        