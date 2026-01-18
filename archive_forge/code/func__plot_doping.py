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
def _plot_doping(self, plt, temp) -> None:
    if len(self._bz.doping) != 0:
        limit = 2210000000000000.0
        plt.axvline(self._bz.mu_doping['n'][temp][0], linewidth=3.0, linestyle='--')
        plt.text(self._bz.mu_doping['n'][temp][0] + 0.01, limit, f'$n$=10^{{{math.log10(self._bz.doping['n'][0])}}}$', color='b')
        plt.axvline(self._bz.mu_doping['n'][temp][-1], linewidth=3.0, linestyle='--')
        plt.text(self._bz.mu_doping['n'][temp][-1] + 0.01, limit, f'$n$=10^{{{math.log10(self._bz.doping['n'][-1])}}}$', color='b')
        plt.axvline(self._bz.mu_doping['p'][temp][0], linewidth=3.0, linestyle='--')
        plt.text(self._bz.mu_doping['p'][temp][0] + 0.01, limit, f'$p$=10^{{{math.log10(self._bz.doping['p'][0])}}}$', color='b')
        plt.axvline(self._bz.mu_doping['p'][temp][-1], linewidth=3.0, linestyle='--')
        plt.text(self._bz.mu_doping['p'][temp][-1] + 0.01, limit, f'$p$=10^{{{math.log10(self._bz.doping['p'][-1])}}}$', color='b')