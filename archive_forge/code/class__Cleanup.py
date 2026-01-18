import atexit
import os
import shutil
import tempfile
import weakref
from fastimport.reftracker import RefTracker
from ... import lru_cache, trace
from . import branch_mapper
from .helpers import single_plural
class _Cleanup:
    """This class makes sure we clean up when CacheManager goes away.

    We use a helper class to ensure that we are never in a refcycle.
    """

    def __init__(self, disk_blobs):
        self.disk_blobs = disk_blobs
        self.tempdir = None
        self.small_blobs = None

    def __del__(self):
        self.finalize()

    def finalize(self):
        if self.disk_blobs is not None:
            for info in self.disk_blobs.values():
                if info[-1] is not None:
                    os.unlink(info[-1])
            self.disk_blobs = None
        if self.small_blobs is not None:
            self.small_blobs.close()
            self.small_blobs = None
        if self.tempdir is not None:
            shutil.rmtree(self.tempdir)