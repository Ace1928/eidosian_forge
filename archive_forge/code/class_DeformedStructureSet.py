from __future__ import annotations
import collections
import itertools
from typing import TYPE_CHECKING, Literal
import numpy as np
import scipy
from pymatgen.core.lattice import Lattice
from pymatgen.core.tensors import SquareTensor, symmetry_reduce
class DeformedStructureSet(collections.abc.Sequence):
    """
    class that generates a set of independently deformed structures that
    can be used to calculate linear stress-strain response.
    """

    def __init__(self, structure: Structure, norm_strains: Sequence[float]=(-0.01, -0.005, 0.005, 0.01), shear_strains: Sequence[float]=(-0.06, -0.03, 0.03, 0.06), symmetry=False) -> None:
        """
        Construct the deformed geometries of a structure. Generates m + n deformed structures
        according to the supplied parameters.

        Args:
            structure (Structure): structure to undergo deformation
            norm_strains (list of floats): strain values to apply
                to each normal mode. Defaults to (-0.01, -0.005, 0.005, 0.01).
            shear_strains (list of floats): strain values to apply
                to each shear mode. Defaults to (-0.06, -0.03, 0.03, 0.06).
            symmetry (bool): whether or not to use symmetry reduction.
        """
        self.undeformed_structure = structure
        self.deformations: list[Deformation] = []
        self.def_structs: list[Structure] = []
        for ind in [(0, 0), (1, 1), (2, 2)]:
            for amount in norm_strains:
                strain = Strain.from_index_amount(ind, amount)
                self.deformations.append(strain.get_deformation_matrix())
        for ind in [(0, 1), (0, 2), (1, 2)]:
            for amount in shear_strains:
                strain = Strain.from_index_amount(ind, amount)
                self.deformations.append(strain.get_deformation_matrix())
        if symmetry:
            self.sym_dict = symmetry_reduce(self.deformations, structure)
            self.deformations = list(self.sym_dict)
        self.deformed_structures = [defo.apply_to_structure(structure) for defo in self.deformations]

    def __iter__(self):
        return iter(self.deformed_structures)

    def __len__(self):
        return len(self.deformed_structures)

    def __getitem__(self, ind):
        return self.deformed_structures[ind]