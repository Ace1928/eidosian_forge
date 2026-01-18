from .db_utilities import decode_torsion, decode_matrices, db_hash
from .sage_helper import _within_sage
from spherogram.codecs import DTcodec
import sys
import sqlite3
import re
import random
import importlib
import collections
def _one_manifold(self, name, M):
    """
        Inflates the given empty Manifold with the table manifold
        with the specified name.
        """
    if hasattr(self, '_regex'):
        if self._regex.match(name) is None:
            raise KeyError('The manifold %s was not found.' % name)
    cursor = self._cursor.execute(self._select + "where name='" + name + "'")
    rows = cursor.fetchall()
    if len(rows) != 1:
        raise KeyError('The manifold %s was not found.' % name)
    return self._manifold_factory(rows[0], M)