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
def _get_matplotlib_figure(self) -> plt.Figure:
    """Returns a matplotlib figure of reaction kinks diagram."""
    ax = pretty_plot(8, 5)
    plt.xlim([-0.05, 1.05])
    kinks = list(zip(*self.get_kinks()))
    _, x, energy, reactions, _ = kinks
    plt.plot(x, energy, 'o-', markersize=8, c='navy', zorder=1)
    plt.scatter(self.minimum[0], self.minimum[1], marker='*', c='red', s=400, zorder=2)
    for x_coord, y_coord, rxn in zip(x, energy, reactions):
        products = ', '.join([latexify(p.reduced_formula) for p in rxn.products if not np.isclose(rxn.get_coeff(p), 0)])
        plt.annotate(products, xy=(x_coord, y_coord), xytext=(10, -30), textcoords='offset points', ha='right', va='bottom', arrowprops={'arrowstyle': '->', 'connectionstyle': 'arc3,rad=0'})
    plt.ylabel(f'Energy (eV/{('atom' if self.norm else 'f.u.')})')
    plt.xlabel(self._get_xaxis_title())
    plt.ylim(self.minimum[1] + 0.05 * self.minimum[1])
    fig = ax.figure
    plt.close(fig)
    return fig