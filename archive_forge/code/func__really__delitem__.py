import time
import logging
import datetime
import itertools
import functools
import threading
from pyzor.engines.common import *
def _really__delitem__(self, key, db=None):
    """__delitem__ without the exception handling."""
    c = db.cursor()
    try:
        c.execute('DELETE FROM %s WHERE digest=%%s' % self.table_name, (key,))
    finally:
        c.close()