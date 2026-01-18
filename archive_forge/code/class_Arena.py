import bisect
from collections import defaultdict
import mmap
import os
import sys
import tempfile
import threading
from .context import reduction, assert_spawning
from . import util
class Arena(object):
    """
        A shared memory area backed by a temporary file (POSIX).
        """
    if sys.platform == 'linux':
        _dir_candidates = ['/dev/shm']
    else:
        _dir_candidates = []

    def __init__(self, size, fd=-1):
        self.size = size
        self.fd = fd
        if fd == -1:
            self.fd, name = tempfile.mkstemp(prefix='pym-%d-' % os.getpid(), dir=self._choose_dir(size))
            os.unlink(name)
            util.Finalize(self, os.close, (self.fd,))
            os.ftruncate(self.fd, size)
        self.buffer = mmap.mmap(self.fd, self.size)

    def _choose_dir(self, size):
        for d in self._dir_candidates:
            st = os.statvfs(d)
            if st.f_bavail * st.f_frsize >= size:
                return d
        return util.get_temp_dir()