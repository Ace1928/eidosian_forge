from .db_utilities import decode_torsion, decode_matrices, db_hash
from .sage_helper import _within_sage
from spherogram.codecs import DTcodec
import sys
import sqlite3
import re
import random
import importlib
import collections
def _manifold_factory(self, row, M=None):
    """
        Factory for "select name, triangulation" queries.
        Returns a Manifold.
        """
    if M is None:
        M = Manifold('empty')
    m = split_filling_info.match(row[1])
    isosig = m.group(1)
    M._from_isosig(isosig)
    fillings = eval('[' + m.group(2).replace(')(', '),(') + ']', {})
    if fillings:
        M.dehn_fill(fillings)
    self._finalize(M, row)
    return M