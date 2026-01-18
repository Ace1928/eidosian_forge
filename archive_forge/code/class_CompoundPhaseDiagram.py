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
class CompoundPhaseDiagram(PhaseDiagram):
    """
    Generates phase diagrams from compounds as terminations instead of
    elements.
    """
    amount_tol = 1e-05

    def __init__(self, entries, terminal_compositions, normalize_terminal_compositions=True):
        """
        Initializes a CompoundPhaseDiagram.

        Args:
            entries ([PDEntry]): Sequence of input entries. For example,
               if you want a Li2O-P2O5 phase diagram, you might have all
               Li-P-O entries as an input.
            terminal_compositions (list[Composition]): Terminal compositions of
                phase space. In the Li2O-P2O5 example, these will be the
                Li2O and P2O5 compositions.
            normalize_terminal_compositions (bool): Whether to normalize the
                terminal compositions to a per atom basis. If normalized,
                the energy above hulls will be consistent
                for comparison across systems. Non-normalized terminals are
                more intuitive in terms of compositional breakdowns.
        """
        self.original_entries = entries
        self.terminal_compositions = terminal_compositions
        self.normalize_terminals = normalize_terminal_compositions
        p_entries, species_mapping = self.transform_entries(entries, terminal_compositions)
        self.species_mapping = species_mapping
        super().__init__(p_entries, elements=species_mapping.values())

    def transform_entries(self, entries, terminal_compositions):
        """
        Method to transform all entries to the composition coordinate in the
        terminal compositions. If the entry does not fall within the space
        defined by the terminal compositions, they are excluded. For example,
        Li3PO4 is mapped into a Li2O:1.5, P2O5:0.5 composition. The terminal
        compositions are represented by DummySpecies.

        Args:
            entries: Sequence of all input entries
            terminal_compositions: Terminal compositions of phase space.

        Returns:
            Sequence of TransformedPDEntries falling within the phase space.
        """
        new_entries = []
        if self.normalize_terminals:
            terminal_compositions = [c.fractional_composition for c in terminal_compositions]
        sp_mapping = {}
        for idx, comp in enumerate(terminal_compositions):
            sp_mapping[comp] = DummySpecies('X' + chr(102 + idx))
        for entry in entries:
            if getattr(entry, 'attribute', None) is None:
                entry.attribute = getattr(entry, 'entry_id', None)
            try:
                transformed_entry = TransformedPDEntry(entry, sp_mapping)
                new_entries.append(transformed_entry)
            except ReactionError:
                pass
            except TransformedPDEntryError:
                pass
        return (new_entries, sp_mapping)

    def as_dict(self):
        """
        Returns:
            MSONable dictionary representation of CompoundPhaseDiagram.
        """
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'original_entries': [entry.as_dict() for entry in self.original_entries], 'terminal_compositions': [c.as_dict() for c in self.terminal_compositions], 'normalize_terminal_compositions': self.normalize_terminals}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): dictionary representation of CompoundPhaseDiagram.

        Returns:
            CompoundPhaseDiagram
        """
        entries = MontyDecoder().process_decoded(dct['original_entries'])
        terminal_compositions = MontyDecoder().process_decoded(dct['terminal_compositions'])
        return cls(entries, terminal_compositions, dct['normalize_terminal_compositions'])