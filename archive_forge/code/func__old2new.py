import json
import numbers
import os
import sqlite3
import sys
from contextlib import contextmanager
import numpy as np
import ase.io.jsonio
from ase.data import atomic_numbers
from ase.calculators.calculator import all_properties
from ase.db.row import AtomsRow
from ase.db.core import (Database, ops, now, lock, invop, parse_selection,
from ase.parallel import parallel_function
def _old2new(self, values):
    if self.type == 'postgresql':
        assert self.version >= 8, 'Your db-version is too old!'
    assert self.version >= 4, 'Your db-file is too old!'
    if self.version < 5:
        pass
    if self.version < 6:
        m = values[23]
        if m is not None and (not isinstance(m, float)):
            magmom = float(self.deblob(m, shape=()))
            values = values[:23] + (magmom,) + values[24:]
    return values