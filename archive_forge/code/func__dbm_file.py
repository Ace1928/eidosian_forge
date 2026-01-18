from contextlib import contextmanager
import dbm
import os
import threading
from ..api import BytesBackend
from ..api import NO_VALUE
from ... import util
@contextmanager
def _dbm_file(self, write):
    with self._use_rw_lock(write):
        with dbm.open(self.filename, 'w' if write else 'r') as dbm_obj:
            yield dbm_obj