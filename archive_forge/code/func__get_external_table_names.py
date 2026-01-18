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
def _get_external_table_names(self, db_con=None):
    """Return a list with the external table names."""
    sql = "SELECT value FROM information WHERE name='external_table_name'"
    with self.managed_connection() as con:
        cur = con.cursor()
        cur.execute(sql)
        ext_tab_names = [x[0] for x in cur.fetchall()]
    return ext_tab_names