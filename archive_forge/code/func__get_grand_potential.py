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
def _get_grand_potential(self, composition: Composition) -> float:
    """
        Computes the grand potential Phi at a given composition and
        chemical potential(s).

        Args:
            composition: Composition object.

        Returns:
            Grand potential at a given composition at chemical potential(s).
        """
    if self.use_hull_energy:
        grand_potential = self.pd_non_grand.get_hull_energy(composition)
    else:
        grand_potential = self._get_entry_energy(self.pd_non_grand, composition)
    grand_potential -= sum((composition[e] * mu for e, mu in self.pd.chempots.items()))
    if self.norm:
        grand_potential /= sum((composition[el] for el in composition if el not in self.pd.chempots))
    return grand_potential