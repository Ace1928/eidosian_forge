from __future__ import annotations
import os
import warnings
import numpy as np
from monty.serialization import loadfn
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
class RLSVolumePredictor:
    """
    Reference lattice scaling (RLS) scheme that predicts the volume of a
    structure based on a known crystal structure.
    """

    def __init__(self, check_isostructural=True, radii_type='ionic-atomic', use_bv=True):
        """
        Args:
            check_isostructural: Whether to test that the two structures are
                isostructural. This algo works best for isostructural compounds.
                Defaults to True.
            radii_type (str): Types of radii to use. You can specify "ionic"
                (only uses ionic radii), "atomic" (only uses atomic radii) or
                "ionic-atomic" (uses either ionic or atomic radii, with a
                preference for ionic where possible).
            use_bv (bool): Whether to use BVAnalyzer to determine oxidation
                states if not present.
        """
        self.check_isostructural = check_isostructural
        self.radii_type = radii_type
        self.use_bv = use_bv

    def predict(self, structure: Structure, ref_structure):
        """
        Given a structure, returns the predicted volume.

        Args:
            structure (Structure): structure w/unknown volume
            ref_structure (Structure): A reference structure with a similar
                structure but different species.

        Returns:
            a float value of the predicted volume
        """
        if self.check_isostructural:
            matcher = StructureMatcher()
            mapping = matcher.get_best_electronegativity_anonymous_mapping(structure, ref_structure)
            if mapping is None:
                raise ValueError('Input structures do not match!')
        if 'ionic' in self.radii_type:
            try:
                if not _is_ox(structure) and self.use_bv:
                    a = BVAnalyzer()
                    structure = a.get_oxi_state_decorated_structure(structure)
                if not _is_ox(ref_structure) and self.use_bv:
                    a = BVAnalyzer()
                    ref_structure = a.get_oxi_state_decorated_structure(ref_structure)
                comp = structure.composition
                ref_comp = ref_structure.composition
                if any((k.ionic_radius is None for k in list(comp))) or any((k.ionic_radius is None for k in list(ref_comp))):
                    raise ValueError('Not all the ionic radii are available!')
                numerator = 0
                denominator = 0
                for k, v in comp.items():
                    numerator += k.ionic_radius * v ** (1 / 3)
                for k, v in ref_comp.items():
                    denominator += k.ionic_radius * v ** (1 / 3)
                return ref_structure.volume * (numerator / denominator) ** 3
            except Exception:
                warnings.warn('Exception occurred. Will attempt atomic radii.')
        if 'atomic' in self.radii_type:
            comp = structure.composition
            ref_comp = ref_structure.composition
            numerator = 0
            denominator = 0
            for k, v in comp.items():
                numerator += k.atomic_radius * v ** (1 / 3)
            for k, v in ref_comp.items():
                denominator += k.atomic_radius * v ** (1 / 3)
            return ref_structure.volume * (numerator / denominator) ** 3
        raise ValueError('Cannot find volume scaling based on radii choices specified!')

    def get_predicted_structure(self, structure: Structure, ref_structure):
        """
        Given a structure, returns back the structure scaled to predicted
        volume.

        Args:
            structure (Structure): structure w/unknown volume
            ref_structure (Structure): A reference structure with a similar
                structure but different species.

        Returns:
            a Structure object with predicted volume
        """
        new_structure = structure.copy()
        new_structure.scale_lattice(self.predict(structure, ref_structure))
        return new_structure