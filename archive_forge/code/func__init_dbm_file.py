from contextlib import contextmanager
import dbm
import os
import threading
from ..api import BytesBackend
from ..api import NO_VALUE
from ... import util
def _init_dbm_file(self):
    exists = os.access(self.filename, os.F_OK)
    if not exists:
        for ext in ('db', 'dat', 'pag', 'dir'):
            if os.access(self.filename + os.extsep + ext, os.F_OK):
                exists = True
                break
    if not exists:
        fh = dbm.open(self.filename, 'c')
        fh.close()