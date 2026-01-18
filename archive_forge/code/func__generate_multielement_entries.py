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
def _generate_multielement_entries(self, entries, nproc=None):
    """
        Create entries for multi-element Pourbaix construction.

        This works by finding all possible linear combinations
        of entries that can result in the specified composition
        from the initialized comp_dict.

        Args:
            entries ([PourbaixEntries]): list of Pourbaix entries
                to process into MultiEntries
            nproc (int): number of processes to be used in parallel
                treatment of entry combos
        """
    n_elems = len(self._elt_comp)
    total_comp = Composition(self._elt_comp)
    entry_combos = [itertools.combinations(entries, idx + 1) for idx in range(n_elems)]
    entry_combos = itertools.chain.from_iterable(entry_combos)
    entry_combos = filter(lambda x: total_comp < MultiEntry(x).composition, entry_combos)
    processed_entries = []
    total = sum((comb(len(entries), idx + 1) for idx in range(n_elems)))
    if total > 1000000.0:
        warnings.warn(f'Your Pourbaix diagram includes {total} entries and may take a long time to generate.')
    if nproc is not None:
        func = partial(self.process_multientry, prod_comp=total_comp)
        with Pool(nproc) as proc_pool:
            processed_entries = list(proc_pool.imap(func, entry_combos))
        processed_entries = list(filter(bool, processed_entries))
    else:
        for entry_combo in entry_combos:
            processed_entry = self.process_multientry(entry_combo, total_comp)
            if processed_entry is not None:
                processed_entries.append(processed_entry)
    return processed_entries