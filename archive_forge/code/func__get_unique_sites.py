from __future__ import annotations
import copy
import logging
from ast import literal_eval
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from monty.json import MSONable, jsanitize
from monty.serialization import dumpfn
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.analysis.magnetism import CollinearMagneticStructureAnalyzer, Ordering
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@staticmethod
def _get_unique_sites(structure):
    """
        Get dict that maps site indices to unique identifiers.

        Args:
            structure (Structure): ground state Structure object.

        Returns:
            unique_site_ids (dict): maps tuples of equivalent site indices to a
                unique int identifier
            wyckoff_ids (dict): maps tuples of equivalent site indices to their
                wyckoff symbols
        """
    s0 = CollinearMagneticStructureAnalyzer(structure, make_primitive=False, threshold=0.0).get_nonmagnetic_structure(make_primitive=False)
    if 'wyckoff' in s0.site_properties:
        s0.remove_site_property('wyckoff')
    symm_s0 = SpacegroupAnalyzer(s0).get_symmetrized_structure()
    wyckoff = ['n/a'] * len(symm_s0)
    equivalent_indices = symm_s0.equivalent_indices
    wyckoff_symbols = symm_s0.wyckoff_symbols
    unique_site_ids = {}
    wyckoff_ids = {}
    for idx, (indices, symbol) in enumerate(zip(equivalent_indices, wyckoff_symbols)):
        unique_site_ids[tuple(indices)] = idx
        wyckoff_ids[idx] = symbol
        for index in indices:
            wyckoff[index] = symbol
    return (unique_site_ids, wyckoff_ids)