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
def _get_value_type_of_table(self, cursor, tab_name):
    """Return the expected value name."""
    sql = 'SELECT value FROM information WHERE name=?'
    cursor.execute(sql, (tab_name + '_dtype',))
    return cursor.fetchone()[0]