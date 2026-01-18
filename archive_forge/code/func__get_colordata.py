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
def _get_colordata(bs, elements, bs_projection):
    """Get color data, including projected band structures.

        Args:
            bs: Bandstructure object
            elements: elements (in desired order) for setting to blue, red, green
            bs_projection: None for no projection, "elements" for element projection

        Returns:
            Dictionary representation of color data.
        """
    contribs = {}
    if bs_projection and bs_projection.lower() == 'elements':
        projections = bs.get_projection_on_elements()
    for spin in (Spin.up, Spin.down):
        if spin in bs.bands:
            contribs[spin] = []
            for band_idx in range(bs.nb_bands):
                colors = []
                for k_idx in range(len(bs.kpoints)):
                    if bs_projection and bs_projection.lower() == 'elements':
                        c = [0, 0, 0, 0]
                        projs = projections[spin][band_idx][k_idx]
                        projs = {k: v ** 2 for k, v in projs.items()}
                        total = sum(projs.values())
                        if total > 0:
                            for idx, e in enumerate(elements):
                                c[idx] = math.sqrt(projs[e] / total)
                        c = [c[1], c[2], c[0], c[3]]
                        if len(elements) == 4:
                            c = [(1 - c[0]) * (1 - c[3]), (1 - c[1]) * (1 - c[3]), (1 - c[2]) * (1 - c[3])]
                        else:
                            c = [c[0], c[1], c[2]]
                    else:
                        c = [0, 0, 0] if spin == Spin.up else [0, 0, 1]
                    colors.append(c)
                contribs[spin].append(colors)
            contribs[spin] = np.array(contribs[spin])
    return contribs