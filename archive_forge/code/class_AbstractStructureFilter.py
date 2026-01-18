from __future__ import annotations
import abc
from collections import defaultdict
from typing import TYPE_CHECKING
from monty.json import MSONable
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core import get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class AbstractStructureFilter(MSONable, abc.ABC):
    """AbstractStructureFilter that defines an API to perform testing of
    Structures. Structures that return True to a test are retained during
    transmutation while those that return False are removed.
    """

    @abc.abstractmethod
    def test(self, structure: Structure):
        """Method to execute the test.

        Args:
            structure (Structure): Input structure to test

        Returns:
            bool: Structures that return true are kept in the Transmuter
                object during filtering.
        """
        return