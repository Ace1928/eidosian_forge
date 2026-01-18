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
def _rgb_triangle(ax, r_label, g_label, b_label, loc) -> None:
    """Draw an RGB triangle legend on the desired axis."""
    if loc not in range(1, 11):
        loc = 2
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    inset_ax = inset_axes(ax, width=1, height=1, loc=loc)
    mesh = 35
    x = []
    y = []
    color = []
    for r in range(mesh):
        for g in range(mesh):
            for b in range(mesh):
                if not (r == 0 and b == 0 and (g == 0)):
                    r1 = r / (r + g + b)
                    g1 = g / (r + g + b)
                    b1 = b / (r + g + b)
                    x.append(0.33 * (2.0 * g1 + r1) / (r1 + b1 + g1))
                    y.append(0.33 * np.sqrt(3) * r1 / (r1 + b1 + g1))
                    rc = math.sqrt(r ** 2 / (r ** 2 + g ** 2 + b ** 2))
                    gc = math.sqrt(g ** 2 / (r ** 2 + g ** 2 + b ** 2))
                    bc = math.sqrt(b ** 2 / (r ** 2 + g ** 2 + b ** 2))
                    color.append([rc, gc, bc])
    inset_ax.scatter(x, y, s=7, marker='.', edgecolor=color)
    inset_ax.set_xlim([-0.35, 1.0])
    inset_ax.set_ylim([-0.35, 1.0])
    inset_ax.text(0.7, -0.2, g_label, fontsize=13, family='Times New Roman', color=(0, 0, 0), horizontalalignment='left')
    inset_ax.text(0.325, 0.7, r_label, fontsize=13, family='Times New Roman', color=(0, 0, 0), horizontalalignment='center')
    inset_ax.text(-0.05, -0.2, b_label, fontsize=13, family='Times New Roman', color=(0, 0, 0), horizontalalignment='right')
    inset_ax.axis('off')