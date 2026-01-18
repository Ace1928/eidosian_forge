import bisect
from collections import defaultdict
import mmap
import os
import sys
import tempfile
import threading
from .context import reduction, assert_spawning
from . import util
def _choose_dir(self, size):
    for d in self._dir_candidates:
        st = os.statvfs(d)
        if st.f_bavail * st.f_frsize >= size:
            return d
    return util.get_temp_dir()