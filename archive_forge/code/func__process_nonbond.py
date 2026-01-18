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
def _process_nonbond(self) -> dict:
    pair_df = pd.DataFrame(self.nonbond_coeffs)
    assert self._is_valid(pair_df), 'Invalid nonbond coefficients with rows varying in length'
    n_pair, n_coeff = pair_df.shape
    pair_df.columns = [f'coeff{i}' for i in range(1, n_coeff + 1)]
    n_mass = len(self.mass_info)
    n_comb = int(n_mass * (n_mass + 1) / 2)
    if n_pair == n_mass:
        kw = 'Pair Coeffs'
        pair_df.index = range(1, n_mass + 1)
    elif n_pair == n_comb:
        kw = 'PairIJ Coeffs'
        ids = list(itertools.combinations_with_replacement(range(1, n_mass + 1), 2))
        id_df = pd.DataFrame(ids, columns=['id1', 'id2'])
        pair_df = pd.concat([id_df, pair_df], axis=1)
    else:
        raise ValueError(f'Expecting {n_mass} Pair Coeffs or {n_comb} PairIJ Coeffs for {n_mass} atom types, got {n_pair}')
    return {kw: pair_df}