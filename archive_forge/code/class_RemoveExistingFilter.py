from __future__ import annotations
import abc
from collections import defaultdict
from typing import TYPE_CHECKING
from monty.json import MSONable
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core import get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class RemoveExistingFilter(AbstractStructureFilter):
    """This filter removes structures existing in a given list from the transmuter."""

    def __init__(self, existing_structures, structure_matcher=None, symprec=None):
        """Remove existing structures based on the structure matcher
        and symmetry (if symprec is given).

        Args:
            existing_structures: List of existing structures to compare with
            structure_matcher: Provides a structure matcher to be used for
                structure comparison.
            symprec: The precision in the symmetry finder algorithm if None (
                default value), no symmetry check is performed and only the
                structure matcher is used. A recommended value is 1e-5.
        """
        self.symprec = symprec
        self.structure_list = []
        self.existing_structures = existing_structures
        if isinstance(structure_matcher, dict):
            self.structure_matcher = StructureMatcher.from_dict(structure_matcher)
        else:
            self.structure_matcher = structure_matcher or StructureMatcher(comparator=ElementComparator())

    def test(self, structure: Structure):
        """Method to execute the test.

        Args:
            structure (Structure): Input structure to test

        Returns:
            bool: True if structure is not in existing list.
        """

        def get_sg(s):
            finder = SpacegroupAnalyzer(s, symprec=self.symprec)
            return finder.get_space_group_number()
        for s in self.existing_structures:
            if (self.structure_matcher._comparator.get_hash(structure.composition) == self.structure_matcher._comparator.get_hash(s.composition) and self.symprec is None or get_sg(s) == get_sg(structure)) and self.structure_matcher.fit(s, structure):
                return False
        self.structure_list.append(structure)
        return True

    def as_dict(self):
        """Returns: MSONable dict."""
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'init_args': {'structure_matcher': self.structure_matcher.as_dict()}}