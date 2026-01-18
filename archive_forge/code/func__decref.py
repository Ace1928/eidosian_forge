import atexit
import os
import shutil
import tempfile
import weakref
from fastimport.reftracker import RefTracker
from ... import lru_cache, trace
from . import branch_mapper
from .helpers import single_plural
def _decref(self, id, cache, fn):
    if not self._blob_ref_counts:
        return False
    count = self._blob_ref_counts.get(id, None)
    if count is not None:
        count -= 1
        if count <= 0:
            del cache[id]
            if fn is not None:
                os.unlink(fn)
            del self._blob_ref_counts[id]
            return True
        else:
            self._blob_ref_counts[id] = count
    return False