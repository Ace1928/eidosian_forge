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
def _do_screen(structures, energies):
    """Screen and sort magnetic orderings based on some criteria.

        Prioritize low energy orderings and large, localized magmoms. do_clean should be run first to sanitize inputs.

        Args:
            structures (list): At least three structure objects.
            energies (list): Energies.

        Returns:
            screened_structures (list): Sorted structures.
            screened_energies (list): Sorted energies.
        """
    magmoms = [s.site_properties['magmom'] for s in structures]
    n_below_1ub = [len([m for m in ms if abs(m) < 1]) for ms in magmoms]
    df = pd.DataFrame({'structure': structures, 'energy': energies, 'magmoms': magmoms, 'n_below_1ub': n_below_1ub})
    index = list(df.index)[2:]
    df_high_energy = df.iloc[2:]
    df_high_energy = df_high_energy.sort_values(by='n_below_1ub')
    index = [0, 1, *df_high_energy.index]
    df = df.reindex(index)
    screened_structures = list(df['structure'].values)
    screened_energies = list(df['energy'].values)
    return (screened_structures, screened_energies)