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
class IonEntry(PDEntry):
    """
    Object similar to PDEntry, but contains an Ion object instead of a
    Composition object.

    Attributes:
        name (str): A name for the entry. This is the string shown in the phase diagrams.
            By default, this is the reduced formula for the composition, but can be
            set to some other string for display purposes.
    """

    def __init__(self, ion: Ion, energy: float, name: str | None=None, attribute=None):
        """
        Args:
            ion: Ion object
            energy: Energy for composition.
            name: Optional parameter to name the entry. Defaults to the
                chemical formula.
            attribute: Optional attribute of the entry, e.g., band gap.
        """
        self.ion = ion
        name = name or self.ion.reduced_formula
        super().__init__(composition=ion.composition, energy=energy, name=name, attribute=attribute)

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Returns an IonEntry object from a dict."""
        return cls(Ion.from_dict(dct['ion']), dct['energy'], dct.get('name'), dct.get('attribute'))

    def as_dict(self):
        """Creates a dict of composition, energy, and ion name."""
        return {'ion': self.ion.as_dict(), 'energy': self.energy, 'name': self.name}

    def __repr__(self):
        return f'IonEntry : {self.composition} with energy = {self.energy:.4f}'