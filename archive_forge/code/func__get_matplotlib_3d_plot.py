from __future__ import annotations
import collections
import itertools
import json
import logging
import math
import os
import re
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, no_type_check
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.font_manager import FontProperties
from monty.json import MontyDecoder, MSONable
from scipy import interpolate
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from tqdm import tqdm
from pymatgen.analysis.reaction_calculator import Reaction, ReactionError
from pymatgen.core import DummySpecies, Element, get_el_sp
from pymatgen.core.composition import Composition
from pymatgen.entries import Entry
from pymatgen.util.coord import Simplex, in_coord_list
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
from pymatgen.util.string import htmlify, latexify
@no_type_check
def _get_matplotlib_3d_plot(self, label_stable=True, ax: plt.Axes=None):
    """
        Shows the plot using matplotlib.

        Args:
            label_stable (bool): Whether to label stable compounds.
            ax (plt.Axes): An existing axes object (optional). If not provided, a new one will be created.

        Returns:
            plt.Axes: The axes object with the plot.
        """
    ax = ax or plt.figure().add_subplot(111, projection='3d')
    font = FontProperties(weight='bold', size=13)
    lines, labels, _ = self.pd_plot_data
    count = 1
    newlabels = []
    for x, y, z in lines:
        ax.plot(x, y, z, 'bo-', linewidth=3, markeredgecolor='b', markerfacecolor='r', markersize=10)
    for coords in sorted(labels):
        entry = labels[coords]
        label = entry.name
        if label_stable:
            if len(entry.elements) == 1:
                ax.text(coords[0], coords[1], coords[2], label, fontproperties=font)
            else:
                ax.text(coords[0], coords[1], coords[2], str(count), fontsize=12)
                newlabels.append(f'{count} : {latexify(label)}')
                count += 1
    plt.figtext(0.01, 0.01, '\n'.join(newlabels), fontproperties=font)
    ax.axis('off')
    ax.set(xlim=(-0.1, 0.72), ylim=(0, 0.66), zlim=(0, 0.56))
    return ax