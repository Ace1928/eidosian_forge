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
class ReactionDiagram:
    """
    Analyzes the possible reactions between a pair of compounds, e.g.,
    an electrolyte and an electrode.
    """

    def __init__(self, entry1, entry2, all_entries, tol: float=0.0001, float_fmt='%.4f'):
        """
        Args:
            entry1 (ComputedEntry): Entry for 1st component. Note that
                corrections, if any, must already be pre-applied. This is to
                give flexibility for different kinds of corrections, e.g.,
                if a particular entry is fitted to an experimental data (such
                as EC molecule).
            entry2 (ComputedEntry): Entry for 2nd component. Note that
                corrections must already be pre-applied. This is to
                give flexibility for different kinds of corrections, e.g.,
                if a particular entry is fitted to an experimental data (such
                as EC molecule).
            all_entries ([ComputedEntry]): All other entries to be
                considered in the analysis. Note that corrections, if any,
                must already be pre-applied.
            tol (float): Tolerance to be used to determine validity of reaction. Defaults to 1e-4.
            float_fmt (str): Formatting string to be applied to all floats. Determines
                number of decimal places in reaction string. Defaults to "%.4f".
        """
        elem_set = set()
        for entry in [entry1, entry2]:
            elem_set.update([el.symbol for el in entry.elements])
        elements = tuple(elem_set)
        comp_vec1 = np.array([entry1.composition.get_atomic_fraction(el) for el in elements])
        comp_vec2 = np.array([entry2.composition.get_atomic_fraction(el) for el in elements])
        r1 = entry1.composition.reduced_composition
        r2 = entry2.composition.reduced_composition
        logger.debug(f'{len(all_entries)} total entries.')
        pd = PhaseDiagram([*all_entries, entry1, entry2])
        terminal_formulas = [entry1.reduced_formula, entry2.reduced_formula]
        logger.debug(f'{len(pd.stable_entries)} stable entries')
        logger.debug(f'{len(pd.facets)} facets')
        logger.debug(f'{len(pd.qhull_entries)} qhull_entries')
        rxn_entries = []
        done: list[tuple[float, float]] = []

        def fmt(fl):
            return float_fmt % fl
        for facet in pd.facets:
            for face in itertools.combinations(facet, len(facet) - 1):
                face_entries = [pd.qhull_entries[idx] for idx in face]
                if any((entry.reduced_formula in terminal_formulas for entry in face_entries)):
                    continue
                try:
                    mat = []
                    for entry in face_entries:
                        mat.append([entry.composition.get_atomic_fraction(el) for el in elements])
                    mat.append(comp_vec2 - comp_vec1)
                    matrix = np.array(mat).T
                    coeffs = np.linalg.solve(matrix, comp_vec2)
                    x = coeffs[-1]
                    if all((c >= -tol for c in coeffs)) and abs(sum(coeffs[:-1]) - 1) < tol and (tol < x < 1 - tol):
                        c1 = x / r1.num_atoms
                        c2 = (1 - x) / r2.num_atoms
                        factor = 1 / (c1 + c2)
                        c1 *= factor
                        c2 *= factor
                        if any((np.allclose([c1, c2], cc) for cc in done)):
                            continue
                        done.append((c1, c2))
                        rxn_str = f'{fmt(c1)} {r1.reduced_formula} + {fmt(c2)} {r2.reduced_formula} -> '
                        products = []
                        product_entries = []
                        energy = -(x * entry1.energy_per_atom + (1 - x) * entry2.energy_per_atom)
                        for c, entry in zip(coeffs[:-1], face_entries):
                            if c > tol:
                                redu_comp = entry.composition.reduced_composition
                                products.append(f'{fmt(c / redu_comp.num_atoms * factor)} {redu_comp.reduced_formula}')
                                product_entries.append((c, entry))
                                energy += c * entry.energy_per_atom
                        rxn_str += ' + '.join(products)
                        comp = x * comp_vec1 + (1 - x) * comp_vec2
                        entry = PDEntry(Composition(dict(zip(elements, comp))), energy=energy, attribute=rxn_str)
                        entry.decomposition = product_entries
                        rxn_entries.append(entry)
                except np.linalg.LinAlgError:
                    form_1 = entry1.reduced_formula
                    form_2 = entry2.reduced_formula
                    logger.debug(f'Reactants = {form_1}, {form_2}')
                    logger.debug(f'Products = {', '.join([entry.reduced_formula for entry in face_entries])}')
        rxn_entries = sorted(rxn_entries, key=lambda e: e.name, reverse=True)
        self.entry1 = entry1
        self.entry2 = entry2
        self.rxn_entries = rxn_entries
        self.labels = {}
        for idx, entry in enumerate(rxn_entries, start=1):
            self.labels[str(idx)] = entry.attribute
            entry.name = str(idx)
        self.all_entries = all_entries
        self.pd = pd

    def get_compound_pd(self):
        """
        Get the CompoundPhaseDiagram object, which can then be used for
        plotting.

        Returns:
            CompoundPhaseDiagram
        """
        entry1 = PDEntry(self.entry1.composition, 0)
        entry2 = PDEntry(self.entry2.composition, 0)
        return CompoundPhaseDiagram([*self.rxn_entries, entry1, entry2], [Composition(entry1.reduced_formula), Composition(entry2.reduced_formula)], normalize_terminal_compositions=False)