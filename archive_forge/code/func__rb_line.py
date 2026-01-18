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
def _rb_line(ax, r_label, b_label, loc) -> None:
    if loc not in range(1, 11):
        loc = 2
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    inset_ax = inset_axes(ax, width=1.2, height=0.4, loc=loc)
    x, y, color = ([], [], [])
    for idx in range(1000):
        x.append(idx / 1800.0 + 0.55)
        y.append(0)
        color.append([math.sqrt(c) for c in [1 - (idx / 1000) ** 2, 0, (idx / 1000) ** 2]])
    inset_ax.scatter(x, y, s=250.0, marker='s', c=color)
    inset_ax.set_xlim([-0.1, 1.7])
    inset_ax.text(1.35, 0, b_label, fontsize=13, family='Times New Roman', color=(0, 0, 0), horizontalalignment='left', verticalalignment='center')
    inset_ax.text(0.3, 0, r_label, fontsize=13, family='Times New Roman', color=(0, 0, 0), horizontalalignment='right', verticalalignment='center')
    inset_ax.axis('off')