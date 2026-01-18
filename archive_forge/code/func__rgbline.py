from __future__ import annotations
import copy
import itertools
import logging
import math
import typing
import warnings
from collections import Counter
from typing import TYPE_CHECKING, Literal, cast, no_type_check
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import palettable
import scipy.interpolate as scint
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from monty.dev import requires
from monty.json import jsanitize
from pymatgen.core import Element
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.boltztrap import BoltztrapError
from pymatgen.electronic_structure.core import OrbitalType, Spin
from pymatgen.util.plotting import add_fig_kwargs, get_ax3d_fig, pretty_plot
@staticmethod
def _rgbline(ax, k, e, red, green, blue, alpha=1, linestyles='solid') -> None:
    """An RGB colored line for plotting.
        creation of segments based on:
        http://nbviewer.ipython.org/urls/raw.github.com/dpsanders/matplotlib-examples/master/colorline.ipynb.

        Args:
            ax: matplotlib axis
            k: x-axis data (k-points)
            e: y-axis data (energies)
            red: red data
            green: green data
            blue: blue data
            alpha: alpha values data
            linestyles: linestyle for plot (e.g., "solid" or "dotted").
        """
    pts = np.array([k, e]).T.reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    n_seg = len(k) - 1
    red = [0.5 * (red[i] + red[i + 1]) for i in range(n_seg)]
    green = [0.5 * (green[i] + green[i + 1]) for i in range(n_seg)]
    blue = [0.5 * (blue[i] + blue[i + 1]) for i in range(n_seg)]
    alpha = np.ones(n_seg, float) * alpha
    lc = LineCollection(seg, colors=list(zip(red, green, blue, alpha)), linewidth=2, linestyles=linestyles)
    ax.add_collection(lc)