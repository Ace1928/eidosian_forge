import time
import logging
import datetime
import itertools
import functools
import threading
from pyzor.engines.common import *
def _really__getitem__(self, key, db=None):
    """__getitem__ without the exception handling."""
    c = db.cursor()
    c.execute('SELECT r_count, wl_count, r_entered, r_updated, wl_entered, wl_updated FROM %s WHERE digest=%%s' % self.table_name, (key,))
    try:
        try:
            return Record(*c.fetchone())
        except TypeError:
            raise KeyError()
    finally:
        c.close()