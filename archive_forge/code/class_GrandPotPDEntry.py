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
class GrandPotPDEntry(PDEntry):
    """
    A grand potential pd entry object encompassing all relevant data for phase
    diagrams. Chemical potentials are given as a element-chemical potential
    dict.
    """

    def __init__(self, entry, chempots, name=None):
        """
        Args:
            entry: A PDEntry-like object.
            chempots: Chemical potential specification as {Element: float}.
            name: Optional parameter to name the entry. Defaults to the reduced
                chemical formula of the original entry.
        """
        super().__init__(entry.composition, entry.energy, name or entry.name, getattr(entry, 'attribute', None))
        self.original_entry = entry
        self.original_comp = self._composition
        self.chempots = chempots

    @property
    def composition(self) -> Composition:
        """The composition after removing free species.

        Returns:
            Composition
        """
        return Composition({el: self._composition[el] for el in self._composition.elements if el not in self.chempots})

    @property
    def chemical_energy(self):
        """The chemical energy term mu*N in the grand potential.

        Returns:
            The chemical energy term mu*N in the grand potential
        """
        return sum((self._composition[el] * pot for el, pot in self.chempots.items()))

    @property
    def energy(self):
        """
        Returns:
            The grand potential energy.
        """
        return self._energy - self.chemical_energy

    def __repr__(self):
        output = [f'GrandPotPDEntry with original composition {self.original_entry.composition}, energy = {self.original_entry.energy:.4f}, ', 'chempots = ' + ', '.join((f'mu_{el} = {mu:.4f}' for el, mu in self.chempots.items()))]
        return ''.join(output)

    def as_dict(self):
        """
        Returns:
            MSONable dictionary representation of GrandPotPDEntry.
        """
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'entry': self.original_entry.as_dict(), 'chempots': {el.symbol: u for el, u in self.chempots.items()}, 'name': self.name}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): dictionary representation of GrandPotPDEntry.

        Returns:
            GrandPotPDEntry
        """
        chempots = {Element(symbol): u for symbol, u in dct['chempots'].items()}
        entry = MontyDecoder().process_decoded(dct['entry'])
        return cls(entry, chempots, dct['name'])