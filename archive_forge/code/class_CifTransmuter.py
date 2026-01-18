from __future__ import annotations
import os
import re
from multiprocessing import Pool
from typing import TYPE_CHECKING, Callable
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.io.vasp.sets import MPRelaxSet, VaspInputSet
class CifTransmuter(StandardTransmuter):
    """Generates a Transmuter from a cif string, possibly containing multiple
    structures.
    """

    def __init__(self, cif_string, transformations=None, primitive=True, extend_collection=False):
        """Generates a Transmuter from a cif string, possibly
        containing multiple structures.

        Args:
            cif_string: A string containing a cif or a series of CIFs
            transformations: New transformations to be applied to all
                structures
            primitive: Whether to generate the primitive cell from the cif.
            extend_collection: Whether to use more than one output structure
                from one-to-many transformations. extend_collection can be a
                number, which determines the maximum branching for each
                transformation.
        """
        transformed_structures = []
        lines = cif_string.split('\n')
        structure_data = []
        read_data = False
        for line in lines:
            if re.match('^\\s*data', line):
                structure_data.append([])
                read_data = True
            if read_data:
                structure_data[-1].append(line)
        for data in structure_data:
            trafo_struct = TransformedStructure.from_cif_str('\n'.join(data), [], primitive)
            transformed_structures.append(trafo_struct)
        super().__init__(transformed_structures, transformations, extend_collection)

    @classmethod
    def from_filenames(cls, filenames, transformations=None, primitive=True, extend_collection=False) -> Self:
        """Generates a TransformedStructureCollection from a cif, possibly
        containing multiple structures.

        Args:
            filenames: List of strings of the cif files
            transformations: New transformations to be applied to all
                structures
            primitive: Same meaning as in __init__.
            extend_collection: Same meaning as in __init__.
        """
        cif_files = []
        for filename in filenames:
            with open(filename, encoding='utf-8') as file:
                cif_files.append(file.read())
        return cls('\n'.join(cif_files), transformations, primitive=primitive, extend_collection=extend_collection)