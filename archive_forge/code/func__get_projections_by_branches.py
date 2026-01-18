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
def _get_projections_by_branches(self, dictio):
    proj = self._bs.get_projections_on_elements_and_orbitals(dictio)
    proj_br = []
    for b in self._bs.branches:
        if self._bs.is_spin_polarized:
            proj_br.append({str(Spin.up): [[] for _ in range(self._nb_bands)], str(Spin.down): [[] for _ in range(self._nb_bands)]})
        else:
            proj_br.append({str(Spin.up): [[] for _ in range(self._nb_bands)]})
        for i in range(self._nb_bands):
            for j in range(b['start_index'], b['end_index'] + 1):
                proj_br[-1][str(Spin.up)][i].append({e: {o: proj[Spin.up][i][j][e][o] for o in proj[Spin.up][i][j][e]} for e in proj[Spin.up][i][j]})
        if self._bs.is_spin_polarized:
            for b in self._bs.branches:
                for i in range(self._nb_bands):
                    for j in range(b['start_index'], b['end_index'] + 1):
                        proj_br[-1][str(Spin.down)][i].append({e: {o: proj[Spin.down][i][j][e][o] for o in proj[Spin.down][i][j][e]} for e in proj[Spin.down][i][j]})
    return proj_br