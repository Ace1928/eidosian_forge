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
@due.dcite(Doi('10.1021/cm702327g'), description='Phase Diagram from First Principles Calculations')
@due.dcite(Doi('10.1016/j.elecom.2010.01.010'), description='Thermal stabilities of delithiated olivine MPO4 (M=Fe, Mn) cathodes investigated using first principles calculations')
class GrandPotentialPhaseDiagram(PhaseDiagram):
    """
    A class representing a Grand potential phase diagram. Grand potential phase
    diagrams are essentially phase diagrams that are open to one or more
    components. To construct such phase diagrams, the relevant free energy is
    the grand potential, which can be written as the Legendre transform of the
    Gibbs free energy as follows.

    Grand potential = G - u_X N_X

    The algorithm is based on the work in the following papers:

    1. S. P. Ong, L. Wang, B. Kang, and G. Ceder, Li-Fe-P-O2 Phase Diagram from
       First Principles Calculations. Chem. Mater., 2008, 20(5), 1798-1807.
       doi:10.1021/cm702327g

    2. S. P. Ong, A. Jain, G. Hautier, B. Kang, G. Ceder, Thermal stabilities
       of delithiated olivine MPO4 (M=Fe, Mn) cathodes investigated using first
       principles calculations. Electrochem. Comm., 2010, 12(3), 427-430.
       doi:10.1016/j.elecom.2010.01.010
    """

    def __init__(self, entries, chempots, elements=None, *, computed_data=None):
        """
        Standard constructor for grand potential phase diagram.

        Args:
            entries ([PDEntry]): A list of PDEntry-like objects having an
                energy, energy_per_atom and composition.
            chempots ({Element: float}): Specify the chemical potentials
                of the open elements.
            elements ([Element]): Optional list of elements in the phase
                diagram. If set to None, the elements are determined from
                the entries themselves.
            computed_data (dict): A dict containing pre-computed data. This allows
                PhaseDiagram object to be reconstituted without performing the
                expensive convex hull computation. The dict is the output from the
                PhaseDiagram._compute() method and is stored in PhaseDiagram.computed_data
                when generated for the first time.
        """
        if elements is None:
            elements = {els for entry in entries for els in entry.elements}
        self.chempots = {get_el_sp(el): u for el, u in chempots.items()}
        elements = set(elements) - set(self.chempots)
        all_entries = [GrandPotPDEntry(entry, self.chempots) for entry in entries if len(elements.intersection(entry.elements)) > 0]
        super().__init__(all_entries, elements, computed_data=None)

    def __repr__(self):
        chemsys = '-'.join((el.symbol for el in self.elements))
        chempots = ', '.join((f'mu_{el} = {mu:.4f}' for el, mu in self.chempots.items()))
        output = [f'{chemsys} GrandPotentialPhaseDiagram with chempots = {chempots!r}', f'{len(self.stable_entries)} stable phases: ', ', '.join((entry.name for entry in self.stable_entries))]
        return ''.join(output)

    def as_dict(self):
        """
        Returns:
            MSONable dictionary representation of GrandPotentialPhaseDiagram.
        """
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'all_entries': [entry.as_dict() for entry in self.all_entries], 'chempots': self.chempots, 'elements': [entry.as_dict() for entry in self.elements]}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): dictionary representation of GrandPotentialPhaseDiagram.

        Returns:
            GrandPotentialPhaseDiagram
        """
        entries = MontyDecoder().process_decoded(dct['all_entries'])
        elements = MontyDecoder().process_decoded(dct['elements'])
        return cls(entries, dct['chempots'], elements)