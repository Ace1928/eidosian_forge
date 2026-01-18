from __future__ import annotations
import abc
from collections import defaultdict
from typing import TYPE_CHECKING
from monty.json import MSONable
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core import get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class SpecieProximityFilter(AbstractStructureFilter):
    """This filter removes structures that have certain species that are too close
    together.
    """

    def __init__(self, specie_and_min_dist_dict):
        """
        Args:
            specie_and_min_dist_dict (dict): A species string to float mapping. For
                example, {"Na+": 1} means that all Na+ ions must be at least 1
                Angstrom away from each other. Multiple species criteria can be
                applied. Note that the testing is done based on the actual object
                . If you have a structure with Element, you must use {"Na":1}
                instead to filter based on Element and not Species.
        """
        self.specie_and_min_dist = {get_el_sp(k): v for k, v in specie_and_min_dist_dict.items()}

    def test(self, structure: Structure):
        """Method to execute the test.

        Args:
            structure (Structure): Input structure to test

        Returns:
            bool: True if structure does not contain species within specified distances.
        """
        all_species = set(self.specie_and_min_dist)
        for site in structure:
            species = set(site.species)
            sp_to_test = species.intersection(all_species)
            if sp_to_test:
                max_r = max((self.specie_and_min_dist[sp] for sp in sp_to_test))
                neighbors = structure.get_neighbors(site, max_r)
                for sp in sp_to_test:
                    for nn_site, dist, *_ in neighbors:
                        if sp in nn_site.species and dist < self.specie_and_min_dist[sp]:
                            return False
        return True

    def as_dict(self):
        """Returns: MSONable dict."""
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'init_args': {'specie_and_min_dist_dict': {str(sp): v for sp, v in self.specie_and_min_dist.items()}}}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): Dict representation.

        Returns:
            Filter
        """
        return cls(**dct['init_args'])