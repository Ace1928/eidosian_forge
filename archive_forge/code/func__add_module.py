import os
from collections import defaultdict
from typing import Callable, Dict, Iterable, Sequence, Set, List
from .. import importcompletion
def _add_module(self, path: str) -> None:
    """Add a python module to track changes"""
    path = os.path.abspath(path)
    for suff in importcompletion.SUFFIXES:
        if path.endswith(suff):
            path = path[:-len(suff)]
            break
    dirname = os.path.dirname(path)
    if dirname not in self.dirs:
        self.observer.schedule(self, dirname, recursive=False)
    self.dirs[dirname].add(path)