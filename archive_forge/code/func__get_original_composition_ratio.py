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
def _get_original_composition_ratio(self, reaction):
    """
        Returns the molar mixing ratio between the reactants with ORIGINAL (
        instead of processed) compositions for a reaction.

        Args:
            reaction (Reaction): Reaction object that contains the original
                reactant compositions.

        Returns:
            The molar mixing ratio between the original reactant
            compositions for a reaction.
        """
    if self.c1_original == self.c2_original:
        return 1
    c1_coeff = reaction.get_coeff(self.c1_original) if self.c1_original in reaction.reactants else 0
    c2_coeff = reaction.get_coeff(self.c2_original) if self.c2_original in reaction.reactants else 0
    return c1_coeff * 1 / (c1_coeff + c2_coeff)