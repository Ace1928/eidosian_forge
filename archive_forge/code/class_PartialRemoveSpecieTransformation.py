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
class PartialRemoveSpecieTransformation(AbstractTransformation):
    """Remove fraction of specie from a structure.

    Requires an oxidation state decorated structure for Ewald sum to be
    computed.

    Given that the solution to selecting the right removals is NP-hard, there
    are several algorithms provided with varying degrees of accuracy and speed.
    Please see
    pymatgen.transformations.site_transformations.PartialRemoveSitesTransformation.
    """
    ALGO_FAST = 0
    ALGO_COMPLETE = 1
    ALGO_BEST_FIRST = 2
    ALGO_ENUMERATE = 3

    def __init__(self, specie_to_remove, fraction_to_remove, algo=ALGO_FAST):
        """
        Args:
            specie_to_remove: Species to remove. Must have oxidation state E.g.,
                "Li+"
            fraction_to_remove: Fraction of specie to remove. E.g., 0.5
            algo: This parameter allows you to choose the algorithm to perform
                ordering. Use one of PartialRemoveSpecieTransformation.ALGO_*
                variables to set the algo.
        """
        self.specie_to_remove = specie_to_remove
        self.fraction_to_remove = fraction_to_remove
        self.algo = algo

    def apply_transformation(self, structure: Structure, return_ranked_list: bool | int=False):
        """Apply the transformation.

        Args:
            structure: input structure
            return_ranked_list (bool | int, optional): If return_ranked_list is int, that number of structures

                is returned. If False, only the single lowest energy structure is returned. Defaults to False.

        Returns:
            Depending on returned_ranked list, either a transformed structure
            or a list of dictionaries, where each dictionary is of the form
            {"structure" = .... , "other_arguments"}
            the key "transformation" is reserved for the transformation that
            was actually applied to the structure.
            This transformation is parsed by the alchemy classes for generating
            a more specific transformation history. Any other information will
            be stored in the transformation_parameters dictionary in the
            transmuted structure class.
        """
        sp = get_el_sp(self.specie_to_remove)
        specie_indices = [i for i in range(len(structure)) if structure[i].species == Composition({sp: 1})]
        trans = PartialRemoveSitesTransformation([specie_indices], [self.fraction_to_remove], algo=self.algo)
        return trans.apply_transformation(structure, return_ranked_list)

    @property
    def is_one_to_many(self) -> bool:
        """Returns: True."""
        return True

    def __repr__(self):
        species = self.specie_to_remove
        fraction_to_remove = self.fraction_to_remove
        algo = self.algo
        return f'PartialRemoveSpecieTransformation(species={species!r}, fraction_to_remove={fraction_to_remove!r}, algo={algo!r})'

    @property
    def inverse(self):
        """Returns: None."""
        return