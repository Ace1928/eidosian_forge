from __future__ import annotations
import json
import os
import warnings
from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
from monty.json import MSONable
from pandas import DataFrame
from plotly.graph_objects import Figure, Scatter
from pymatgen.analysis.phase_diagram import GrandPotentialPhaseDiagram, PhaseDiagram
from pymatgen.analysis.reaction_calculator import Reaction
from pymatgen.core.composition import Composition
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
from pymatgen.util.string import htmlify, latexify
@staticmethod
def _get_entry_energy(pd: PhaseDiagram, composition: Composition):
    """
        Finds the lowest entry energy for entries matching the composition.
        Entries with non-negative formation energies are excluded. If no
        entry is found, use the convex hull energy for the composition.

        Args:
            pd: Phase diagram object
            composition: Composition object that the target entry should match

        Returns:
            The lowest entry energy among entries matching the composition.
        """
    candidate = [entry.energy_per_atom for entry in pd.qhull_entries if entry.composition.fractional_composition == composition.fractional_composition]
    if not candidate:
        warnings.warn(f'The reactant {composition.reduced_formula} has no matching entry with negative formation energy, instead convex hull energy for this composition will be used for reaction energy calculation.')
        return pd.get_hull_energy(composition)
    min_entry_energy = min(candidate)
    return min_entry_energy * composition.num_atoms