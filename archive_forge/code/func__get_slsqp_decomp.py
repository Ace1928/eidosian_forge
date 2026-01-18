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
def _get_slsqp_decomp(comp, competing_entries, tols=(1e-08,), maxiter=1000):
    """
    Finds the amounts of competing compositions that minimize the energy of a
    given composition.

    The algorithm is based on the work in the following paper:

    1. Bartel, C., Trewartha, A., Wang, Q., Dunn, A., Jain, A., Ceder, G.,
        A critical examination of compound stability predictions from
        machine-learned formation energies, npj Computational Materials 6, 97 (2020)

    Args:
        comp (Composition): A Composition to analyze
        competing_entries ([PDEntry]): List of entries to consider for decomposition
        tols (list): tolerances to try for SLSQP convergence. Issues observed for
            tol > 1e-7 in the fractional composition (default 1e-8)
        maxiter (int): maximum number of SLSQP iterations

    Returns:
            decomposition as a dict of {PDEntry: amount} where amount
            is the amount of the fractional composition.
    """
    amounts = comp.get_el_amt_dict()
    chemical_space = tuple(amounts)
    b = np.array([amounts[el] for el in chemical_space])
    A_transpose = np.zeros((len(chemical_space), len(competing_entries)))
    for ii, comp_entry in enumerate(competing_entries):
        amounts = comp_entry.composition.get_el_amt_dict()
        for jj, el in enumerate(chemical_space):
            A_transpose[jj, ii] = amounts.get(el, 0)
    b = b / np.sum(b)
    A_transpose = A_transpose / np.sum(A_transpose, axis=0)
    Es = np.array([comp_entry.energy_per_atom for comp_entry in competing_entries])
    molar_constraint = {'type': 'eq', 'fun': lambda x: np.dot(A_transpose, x) - b, 'jac': lambda x: A_transpose}
    options = {'maxiter': maxiter, 'disp': False}
    max_bound = comp.num_atoms
    bounds = [(0, max_bound)] * len(competing_entries)
    x0 = [1 / len(competing_entries)] * len(competing_entries)
    for tol in sorted(tols):
        solution = minimize(fun=lambda x: np.dot(x, Es), x0=x0, method='SLSQP', jac=lambda x: Es, bounds=bounds, constraints=[molar_constraint], tol=tol, options=options)
        if solution.success:
            decomp_amts = solution.x
            return {c: amt for c, amt in zip(competing_entries, decomp_amts) if amt > PhaseDiagram.numerical_tol}
    raise ValueError(f'No valid decomp found for {comp}!')