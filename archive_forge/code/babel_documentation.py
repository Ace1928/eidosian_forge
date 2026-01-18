from __future__ import annotations
import copy
import warnings
from typing import TYPE_CHECKING
from monty.dev import requires
from pymatgen.core.structure import IMolecule, Molecule

        Uses OpenBabel to read a molecule from a string in all supported
        formats.

        Args:
            string_data: String containing molecule data.
            file_format: String specifying any OpenBabel supported formats.

        Returns:
            BabelMolAdaptor object
        