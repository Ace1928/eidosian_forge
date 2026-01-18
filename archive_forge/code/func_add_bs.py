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
def add_bs(self, bs: BandStructureSymmLine | list[BandStructureSymmLine]) -> None:
    """Method to add bands objects to the BSPlotter."""
    if not isinstance(bs, list):
        bs = [bs]
    if self._check_bs_kpath(bs):
        self._bs.extend(bs)
        self._nb_bands.extend([b.nb_bands for b in bs])