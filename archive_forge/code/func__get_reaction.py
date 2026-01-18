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
def _get_reaction(self, x: float) -> Reaction:
    """
        Generates balanced reaction at mixing ratio x : (1-x) for
        self.comp1 : self.comp2.

        Args:
            x (float): Mixing ratio x of reactants, a float between 0 and 1.

        Returns:
            Reaction object.
        """
    mix_comp = self.comp1 * x + self.comp2 * (1 - x)
    decomp = self.pd.get_decomposition(mix_comp)
    reactants = self._get_reactants(x)
    product = [Composition(entry.name) for entry in decomp]
    reaction = Reaction(reactants, product)
    x_original = self._get_original_composition_ratio(reaction)
    if np.isclose(x_original, 1):
        reaction.normalize_to(self.c1_original, x_original)
    else:
        reaction.normalize_to(self.c2_original, 1 - x_original)
    return reaction