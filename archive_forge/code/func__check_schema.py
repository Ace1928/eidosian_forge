from .db_utilities import decode_torsion, decode_matrices, db_hash
from .sage_helper import _within_sage
from spherogram.codecs import DTcodec
import sys
import sqlite3
import re
import random
import importlib
import collections
def _check_schema(self):
    assert self.schema['name'] == 'text' and self.schema['triangulation'] == 'text', 'Not a valid Manifold table.'