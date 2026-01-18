from __future__ import annotations
import fnmatch
import os
import subprocess
import sys
import threading
import time
import typing as t
from itertools import chain
from pathlib import PurePath
from ._internal import _log
def _find_stat_paths(extra_files: set[str], exclude_patterns: set[str]) -> t.Iterable[str]:
    """Find paths for the stat reloader to watch. Returns imported
    module files, Python files under non-system paths. Extra files and
    Python files under extra directories can also be scanned.

    System paths have to be excluded for efficiency. Non-system paths,
    such as a project root or ``sys.path.insert``, should be the paths
    of interest to the user anyway.
    """
    paths = set()
    for path in chain(list(sys.path), extra_files):
        path = os.path.abspath(path)
        if os.path.isfile(path):
            paths.add(path)
            continue
        parent_has_py = {os.path.dirname(path): True}
        for root, dirs, files in os.walk(path):
            if root.startswith(_stat_ignore_scan) or os.path.basename(root) in _ignore_common_dirs:
                dirs.clear()
                continue
            has_py = False
            for name in files:
                if name.endswith(('.py', '.pyc')):
                    has_py = True
                    paths.add(os.path.join(root, name))
            if not (has_py or parent_has_py[os.path.dirname(root)]):
                dirs.clear()
                continue
            parent_has_py[root] = has_py
    paths.update(_iter_module_paths())
    _remove_by_pattern(paths, exclude_patterns)
    return paths