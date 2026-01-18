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
def _get_reactants(self, x: float) -> list[Composition]:
    """Returns a list of relevant reactant compositions given an x coordinate."""
    reactants = super()._get_reactants(x)
    reactants += [Composition(entry.symbol) for entry in self.pd.chempots]
    return reactants