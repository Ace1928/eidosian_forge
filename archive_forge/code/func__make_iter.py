from __future__ import unicode_literals
import typing
import re
from collections import namedtuple
from . import wildcard
from ._repr import make_repr
from .lrucache import LRUCache
from .path import iteratepath
def _make_iter(self, search='breadth', namespaces=None):
    try:
        levels, recursive, re_pattern = _PATTERN_CACHE[self.pattern, self.case_sensitive]
    except KeyError:
        levels, recursive, re_pattern = _translate_glob(self.pattern, case_sensitive=self.case_sensitive)
    for path, info in self.fs.walk.info(path=self.path, namespaces=namespaces or self.namespaces, max_depth=None if recursive else levels, search=search, exclude_dirs=self.exclude_dirs):
        if info.is_dir:
            path += '/'
        if re_pattern.match(path):
            yield GlobMatch(path, info)