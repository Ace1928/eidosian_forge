from __future__ import annotations
import copy
import os
import warnings
from itertools import groupby
import numpy as np
import pandas as pd
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.entries.compatibility import (
from pymatgen.entries.computed_entries import ComputedStructureEntry, ConstantEnergyAdjustment
from pymatgen.entries.entry_tools import EntrySet
def _filter_and_sort_entries(self, entries, verbose=True):
    """Given a single list of entries, separate them by run_type and return two lists, one containing
        only entries of each run_type.
        """
    filtered_entries = []
    for entry in entries:
        entry_id = entry.entry_id
        if not entry.parameters.get('run_type'):
            warnings.warn(f'Entry {entry_id} is missing parameters.run_type! This fieldis required. This entry will be ignored.')
            continue
        run_type = entry.parameters.get('run_type')
        if run_type not in [*self.valid_rtypes_1, *self.valid_rtypes_2]:
            warnings.warn(f'Invalid run_type={run_type!r} for entry {entry_id}. Must be one of {self.valid_rtypes_1 + self.valid_rtypes_2}. This entry will be ignored.')
            continue
        formula = entry.reduced_formula
        if entry_id is None:
            warnings.warn(f'entry_id={entry_id!r} for formula={formula!r}. Unique entry_ids are required for every ComputedStructureEntry. This entry will be ignored.')
            continue
        filtered_entries.append(entry)
    filtered_entry_ids = {e.entry_id for e in filtered_entries}
    if len(filtered_entry_ids) != len(filtered_entries):
        raise ValueError('The provided ComputedStructureEntry do not all have unique entry_ids. Unique entry_ids are required for every ComputedStructureEntry.')
    entries_type_1 = [e for e in filtered_entries if e.parameters['run_type'] in self.valid_rtypes_1]
    entries_type_2 = [e for e in filtered_entries if e.parameters['run_type'] in self.valid_rtypes_2]
    if verbose:
        print(f'Processing {len(entries_type_1)} {self.run_type_1} and {len(entries_type_2)} {self.run_type_2} entries...')
    if self.compat_1:
        entries_type_1 = self.compat_1.process_entries(entries_type_1)
        if verbose:
            print(f'  Processed {len(entries_type_1)} compatible {self.run_type_1} entries with {type(self.compat_1).__name__}')
    entries_type_1 = EntrySet(entries_type_1)
    if self.compat_2:
        entries_type_2 = self.compat_2.process_entries(entries_type_2)
        if verbose:
            print(f'  Processed {len(entries_type_2)} compatible {self.run_type_2} entries with {type(self.compat_2).__name__}')
    entries_type_2 = EntrySet(entries_type_2)
    if len(entries_type_1.chemsys) > 0:
        chemsys = entries_type_1.chemsys
        if not entries_type_2.chemsys <= entries_type_1.chemsys:
            warnings.warn(f'  {self.run_type_2} entries chemical system {entries_type_2.chemsys} is larger than {self.run_type_1} entries chemical system {entries_type_1.chemsys}. Entries outside the {self.run_type_1} chemical system will be discarded')
            entries_type_2 = entries_type_2.get_subset_in_chemsys(chemsys)
    else:
        chemsys = entries_type_2.chemsys
    if verbose:
        print(f'  Entries belong to the {chemsys} chemical system')
    return (list(entries_type_1), list(entries_type_2))