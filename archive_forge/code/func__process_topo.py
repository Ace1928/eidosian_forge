from __future__ import annotations
import itertools
import re
import warnings
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable
from monty.serialization import loadfn
from ruamel.yaml import YAML
from pymatgen.core import Element, Lattice, Molecule, Structure
from pymatgen.core.operations import SymmOp
from pymatgen.util.io_utils import clean_lines
def _process_topo(self, kw: str, topo_coeffs: dict) -> tuple[dict, dict]:

    def find_eq_types(label, section) -> list:
        if section.startswith('Improper'):
            label_arr = np.array(label)
            seqs = [[0, 1, 2, 3], [0, 2, 1, 3], [3, 1, 2, 0], [3, 2, 1, 0]]
            return [tuple(label_arr[s]) for s in seqs]
        return [label, label[::-1]]
    main_data, distinct_types = ([], [])
    class2_data: dict = {key: [] for key in topo_coeffs[kw][0] if key in CLASS2_KEYWORDS.get(kw, [])}
    for d in topo_coeffs[kw]:
        main_data.append(d['coeffs'])
        distinct_types.append(d['types'])
        for k in class2_data:
            class2_data[k].append(d[k])
    distinct_types = [set(itertools.chain(*(find_eq_types(t, kw) for t in dt))) for dt in distinct_types]
    type_counts = sum((len(dt) for dt in distinct_types))
    type_union = set.union(*distinct_types)
    assert len(type_union) == type_counts, f'Duplicated items found under different coefficients in {kw}'
    atoms = set(np.ravel(list(itertools.chain(*distinct_types))))
    assert atoms.issubset(self.maps['Atoms']), f'Undefined atom type found in {kw}'
    mapper = {}
    for i, dt in enumerate(distinct_types, start=1):
        for t in dt:
            mapper[t] = i

    def process_data(data) -> pd.DataFrame:
        df = pd.DataFrame(data)
        assert self._is_valid(df), 'Invalid coefficients with rows varying in length'
        n, c = df.shape
        df.columns = [f'coeff{i}' for i in range(1, c + 1)]
        df.index = range(1, n + 1)
        return df
    all_data = {kw: process_data(main_data)}
    if class2_data:
        all_data.update({k: process_data(v) for k, v in class2_data.items()})
    return (all_data, {kw[:-7] + 's': mapper})