from __future__ import annotations
import abc
from collections import defaultdict
from typing import TYPE_CHECKING
from monty.json import MSONable
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core import get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class ChargeBalanceFilter(AbstractStructureFilter):
    """This filter removes structures that are not charge balanced from the
    transmuter. This only works if the structure is oxidation state
    decorated, as structures with only elemental sites are automatically
    assumed to have net charge of 0.
    """

    def __init__(self):
        """No args required."""

    def test(self, structure: Structure):
        """Method to execute the test.

        Args:
            structure (Structure): Input structure to test

        Returns:
            bool: True if structure is neutral.
        """
        return structure.charge == 0.0