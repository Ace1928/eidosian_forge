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
def _do_cleanup(structures, energies):
    """Sanitize input structures and energies.

        Takes magnetic structures and performs the following operations
        - Erases nonmagnetic ions and gives all ions ['magmom'] site prop
        - Converts total energies -> energy / magnetic ion
        - Checks for duplicate/degenerate orderings
        - Sorts by energy

        Args:
            structures (list): Structure objects with magmoms.
            energies (list): Corresponding energies.

        Returns:
            ordered_structures (list): Sanitized structures.
            ordered_energies (list): Sorted energies.
        """
    ordered_structures = [CollinearMagneticStructureAnalyzer(s, make_primitive=False, threshold=0.0).get_structure_with_only_magnetic_atoms(make_primitive=False) for s in structures]
    energies = [e / len(s) for e, s in zip(energies, ordered_structures)]
    remove_list = []
    e_tol = 6
    for idx, energy in enumerate(energies):
        energy = round(energy, e_tol)
        if idx not in remove_list:
            for i_check, e_check in enumerate(energies):
                e_check = round(e_check, e_tol)
                if idx != i_check and i_check not in remove_list and (energy == e_check):
                    remove_list.append(i_check)
    if remove_list:
        ordered_structures = [struct for idx, struct in enumerate(ordered_structures) if idx not in remove_list]
        energies = [energy for idx, energy in enumerate(energies) if idx not in remove_list]
    ordered_structures = [s for _, s in sorted(zip(energies, ordered_structures), reverse=False)]
    ordered_energies = sorted(energies, reverse=False)
    return (ordered_structures, ordered_energies)