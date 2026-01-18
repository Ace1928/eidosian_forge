import atexit
import os
import shutil
import tempfile
import weakref
from fastimport.reftracker import RefTracker
from ... import lru_cache, trace
from . import branch_mapper
from .helpers import single_plural
def _flush_blobs_to_disk(self):
    blobs = list(self._sticky_blobs)
    sticky_blobs = self._sticky_blobs
    total_blobs = len(sticky_blobs)
    blobs.sort(key=lambda k: len(sticky_blobs[k]))
    if self._tempdir is None:
        tempdir = tempfile.mkdtemp(prefix='fastimport_blobs-')
        self._tempdir = tempdir
        self._cleanup.tempdir = self._tempdir
        self._cleanup.small_blobs = tempfile.TemporaryFile(prefix='small-blobs-', dir=self._tempdir)
        small_blob_ref = weakref.ref(self._cleanup.small_blobs)

        def exit_cleanup():
            small_blob = small_blob_ref()
            if small_blob is not None:
                small_blob.close()
            shutil.rmtree(tempdir, ignore_errors=True)
        atexit.register(exit_cleanup)
    count = 0
    bytes = 0
    n_small_bytes = 0
    while self._sticky_memory_bytes > self._sticky_flushed_size:
        id = blobs.pop()
        blob = self._sticky_blobs.pop(id)
        n_bytes = len(blob)
        self._sticky_memory_bytes -= n_bytes
        if n_bytes < self._small_blob_threshold:
            f = self._cleanup.small_blobs
            f.seek(0, os.SEEK_END)
            self._disk_blobs[id] = (f.tell(), n_bytes, None)
            f.write(blob)
            n_small_bytes += n_bytes
        else:
            fd, name = tempfile.mkstemp(prefix='blob-', dir=self._tempdir)
            os.write(fd, blob)
            os.close(fd)
            self._disk_blobs[id] = (0, n_bytes, name)
        bytes += n_bytes
        del blob
        count += 1
    trace.note('flushed %d/%d blobs w/ %.1fMB (%.1fMB small) to disk' % (count, total_blobs, bytes / 1024.0 / 1024, n_small_bytes / 1024.0 / 1024))