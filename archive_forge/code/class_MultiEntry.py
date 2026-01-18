from __future__ import annotations
import itertools
import logging
import re
import warnings
from copy import deepcopy
from functools import cmp_to_key, partial
from multiprocessing import Pool
from typing import TYPE_CHECKING, Any, no_type_check
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.special import comb
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.analysis.reaction_calculator import Reaction, ReactionError
from pymatgen.core import Composition, Element
from pymatgen.core.ion import Ion
from pymatgen.entries.compatibility import MU_H2O
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.util.coord import Simplex
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
from pymatgen.util.string import Stringify
class MultiEntry(PourbaixEntry):
    """
    PourbaixEntry-like object for constructing multi-elemental Pourbaix diagrams.
    """

    def __init__(self, entry_list, weights=None):
        """
        Initializes a MultiEntry.

        Args:
            entry_list ([PourbaixEntry]): List of component PourbaixEntries
            weights ([float]): Weights associated with each entry. Default is None
        """
        self.weights = weights or [1.0] * len(entry_list)
        self.entry_list = entry_list

    def __getattr__(self, attr):
        """
        Because most of the attributes here are just weighted averages of the entry_list,
        we save some space by having a set of conditionals to define the attributes.
        """
        if attr in ['energy', 'npH', 'nH2O', 'nPhi', 'conc_term', 'composition', 'uncorrected_energy', 'elements']:
            start = Composition() if attr == 'composition' else 0
            weighted_values = (getattr(entry, attr) * weight for entry, weight in zip(self.entry_list, self.weights))
            return sum(weighted_values, start)
        if attr in ['entry_id', 'phase_type']:
            return [getattr(entry, attr) for entry in self.entry_list]
        return self.__getattribute__(attr)

    @property
    def name(self):
        """MultiEntry name, i. e. the name of each entry joined by ' + '."""
        return ' + '.join((entry.name for entry in self.entry_list))

    def __repr__(self):
        energy, npH, nPhi, nH2O, entry_id = (self.energy, self.npH, self.nPhi, self.nH2O, self.entry_id)
        cls_name, species = (type(self).__name__, self.name)
        return f'Pourbaix{cls_name}(energy={energy:.4f}, npH={npH!r}, nPhi={nPhi!r}, nH2O={nH2O!r}, entry_id={entry_id!r}, species={species!r})'

    def as_dict(self):
        """Returns: MSONable dict."""
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'entry_list': [entry.as_dict() for entry in self.entry_list], 'weights': self.weights}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): Dict representation.

        Returns:
            MultiEntry
        """
        entry_list = [PourbaixEntry.from_dict(entry) for entry in dct.get('entry_list', ())]
        return cls(entry_list, dct.get('weights'))