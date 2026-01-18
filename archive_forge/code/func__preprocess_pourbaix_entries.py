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
def _preprocess_pourbaix_entries(self, entries, nproc=None):
    """
        Generates multi-entries for Pourbaix diagram.

        Args:
            entries ([PourbaixEntry]): list of PourbaixEntries to preprocess
                into MultiEntries
            nproc (int): number of processes to be used in parallel
                treatment of entry combos

        Returns:
            list[MultiEntry]: stable MultiEntry candidates
        """
    tot_comp = Composition(self._elt_comp)
    min_entries, valid_facets = self._get_hull_in_nph_nphi_space(entries)
    combos = []
    for facet in valid_facets:
        for idx in range(1, self.dim + 2):
            these_combos = []
            for combo in itertools.combinations(facet, idx):
                these_entries = [min_entries[i] for i in combo]
                these_combos.append(frozenset(these_entries))
            combos.append(these_combos)
    all_combos = set(itertools.chain.from_iterable(combos))
    list_combos = []
    for idx in all_combos:
        list_combos.append(list(idx))
    all_combos = list_combos
    multi_entries = []
    if nproc is not None:
        func = partial(self.process_multientry, prod_comp=tot_comp)
        with Pool(nproc) as proc_pool:
            multi_entries = list(proc_pool.imap(func, all_combos))
        multi_entries = list(filter(bool, multi_entries))
    else:
        for combo in all_combos:
            multi_entry = self.process_multientry(combo, prod_comp=tot_comp)
            if multi_entry:
                multi_entries.append(multi_entry)
    return multi_entries