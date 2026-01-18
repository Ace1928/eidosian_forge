from __future__ import unicode_literals
import typing
import re
from collections import namedtuple
from . import wildcard
from ._repr import make_repr
from .lrucache import LRUCache
from .path import iteratepath
def _translate_glob(pattern, case_sensitive=True):
    levels = 0
    recursive = False
    re_patterns = ['']
    for component in iteratepath(pattern):
        if component == '**':
            re_patterns.append('.*/?')
            recursive = True
        else:
            re_patterns.append('/' + wildcard._translate(component, case_sensitive=case_sensitive))
        levels += 1
    re_glob = '(?ms)^' + ''.join(re_patterns) + ('/$' if pattern.endswith('/') else '$')
    return (levels, recursive, re.compile(re_glob, 0 if case_sensitive else re.IGNORECASE))