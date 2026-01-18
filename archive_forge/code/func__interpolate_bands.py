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
def _interpolate_bands(distances, energies, smooth_tol=0, smooth_k=3, smooth_np=100):
    """Method that interpolates the provided energies using B-splines as
        implemented in scipy.interpolate. Distances and energies has to provided
        already split into pieces (branches work good, for longer segments
        the interpolation may fail).

        Interpolation failure can be caused by trying to fit an entire
        band with one spline rather than fitting with piecewise splines
        (splines are ill-suited to fit discontinuities).

        The number of splines used to fit a band is determined by the
        number of branches (high symmetry lines) defined in the
        BandStructureSymmLine object (see BandStructureSymmLine._branches).
        """
    int_energies, int_distances = ([], [])
    smooth_k_orig = smooth_k
    for dist, ene in zip(distances, energies):
        br_en = []
        warning_nan = f'WARNING! Distance / branch, band cannot be interpolated. See full warning in source. If this is not a mistake, try increasing smooth_tol. Current smooth_tol={smooth_tol!r}.'
        warning_m_fewer_k = f'The number of points (m) has to be higher then the order (k) of the splines. In this branch {len(dist)} points are found, while k is set to {smooth_k}. Smooth_k will be reduced to {smooth_k - 1} for this branch.'
        if len(dist) in (2, 3):
            smooth_k = len(dist) - 1
            warnings.warn(warning_m_fewer_k)
        elif len(dist) == 1:
            warnings.warn('Skipping single point branch')
            continue
        int_distances.append(np.linspace(dist[0], dist[-1], smooth_np))
        for ien in ene:
            tck = scint.splrep(dist, ien, s=smooth_tol, k=smooth_k)
            br_en.append(scint.splev(int_distances[-1], tck))
        smooth_k = smooth_k_orig
        int_energies.append(np.vstack(br_en))
        if np.any(np.isnan(int_energies[-1])):
            warnings.warn(warning_nan)
    return (int_distances, int_energies)