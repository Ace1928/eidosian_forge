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
class ScaleToRelaxedTransformation(AbstractTransformation):
    """Takes the unrelaxed and relaxed structure and applies its site and volume
    relaxation to a structurally similar structures (e.g. bulk: NaCl and PbTe
    (rock-salt), slab: Sc(10-10) and Mg(10-10) (hcp), GB: Mo(001) sigma 5 GB,
    Fe(001) sigma 5). Useful for finding an initial guess of a set of similar
    structures closer to its most relaxed state.
    """

    def __init__(self, unrelaxed_structure, relaxed_structure, species_map=None):
        """
        Args:
            unrelaxed_structure (Structure): Initial, unrelaxed structure
            relaxed_structure (Structure): Relaxed structure
            species_map (dict): A dict or list of tuples containing the species mapping in
                string-string pairs. The first species corresponds to the relaxed
                structure while the second corresponds to the species in the
                structure to be scaled. E.g., {"Li":"Na"} or [("Fe2+","Mn2+")].
                Multiple substitutions can be done. Overloaded to accept
                sp_and_occu dictionary E.g. {"Si: {"Ge":0.75, "C":0.25}},
                which substitutes a single species with multiple species to
                generate a disordered structure.
        """
        relax_params = list(relaxed_structure.lattice.abc)
        relax_params.extend(relaxed_structure.lattice.angles)
        unrelax_params = list(unrelaxed_structure.lattice.abc)
        unrelax_params.extend(unrelaxed_structure.lattice.angles)
        self.params_percent_change = [relax_params[idx] / unrelax_params[idx] for idx in range(len(relax_params))]
        self.unrelaxed_structure = unrelaxed_structure
        self.relaxed_structure = relaxed_structure
        self.species_map = species_map

    def apply_transformation(self, structure):
        """Returns a copy of structure with lattice parameters
        and sites scaled to the same degree as the relaxed_structure.

        Args:
            structure (Structure): A structurally similar structure in
                regards to crystal and site positions.
        """
        if self.species_map is None:
            match = StructureMatcher()
            s_map = match.get_best_electronegativity_anonymous_mapping(self.unrelaxed_structure, structure)
        else:
            s_map = self.species_map
        params = list(structure.lattice.abc)
        params.extend(structure.lattice.angles)
        new_lattice = Lattice.from_parameters(*(param * self.params_percent_change[idx] for idx, param in enumerate(params)))
        species, frac_coords = ([], [])
        for site in self.relaxed_structure:
            species.append(s_map[site.specie])
            frac_coords.append(site.frac_coords)
        return Structure(new_lattice, species, frac_coords)

    def __repr__(self):
        return 'ScaleToRelaxedTransformation'

    @property
    def inverse(self):
        """Returns: None."""
        return

    @property
    def is_one_to_many(self) -> bool:
        """Returns: False."""
        return False