import os
import re
import sys
import copy
import glob
import atexit
import tempfile
import subprocess
import shutil
import multiprocessing
import textwrap
import importlib.util
from threading import local as tlocal
from functools import reduce
import distutils
from distutils.errors import DistutilsError
def _get_hg_revision(self, path):
    """Return path's Mercurial revision number.
        """
    try:
        output = subprocess.check_output(['hg', 'identify', '--num'], cwd=path)
    except (subprocess.CalledProcessError, OSError):
        pass
    else:
        m = re.match(b'(?P<revision>\\d+)', output)
        if m:
            return int(m.group('revision'))
    branch_fn = njoin(path, '.hg', 'branch')
    branch_cache_fn = njoin(path, '.hg', 'branch.cache')
    if os.path.isfile(branch_fn):
        branch0 = None
        with open(branch_fn) as f:
            revision0 = f.read().strip()
        branch_map = {}
        with open(branch_cache_fn) as f:
            for line in f:
                branch1, revision1 = line.split()[:2]
                if revision1 == revision0:
                    branch0 = branch1
                try:
                    revision1 = int(revision1)
                except ValueError:
                    continue
                branch_map[branch1] = revision1
        return branch_map.get(branch0)
    return None