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
def _get_plotly_annotations(x: list[float], y: list[float], reactions: list[Reaction]):
    """Returns dictionary of annotations for the Plotly figure layout."""
    annotations = []
    for x_coord, y_coord, rxn in zip(x, y, reactions):
        products = ', '.join([htmlify(p.reduced_formula) for p in rxn.products if not np.isclose(rxn.get_coeff(p), 0)])
        annotation = {'x': x_coord, 'y': y_coord, 'text': products, 'font': {'size': 18}, 'ax': -25, 'ay': 55}
        annotations.append(annotation)
    return annotations