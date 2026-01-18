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
def add_cohp_dict(self, cohp_dict, key_sort_func=None) -> None:
    """Adds a dictionary of COHPs with an optional sorting function
        for the keys.

        Args:
            cohp_dict: dict of the form {label: Cohp}

            key_sort_func: function used to sort the cohp_dict keys.
        """
    keys = sorted(cohp_dict, key=key_sort_func) if key_sort_func else list(cohp_dict)
    for label in keys:
        self.add_cohp(label, cohp_dict[label])