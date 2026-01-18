import contextlib
import tempfile
from typing import Type
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from . import decorators, errors, hooks, osutils, registry
from . import revision as _mod_revision
from . import trace, transform
from . import transport as _mod_transport
from . import tree as _mod_tree
def _iter_plan(self, blocks, new_a, killed_b, new_b, killed_a):
    last_i = 0
    last_j = 0
    for i, j, n in blocks:
        for a_index in range(last_i, i):
            if a_index in new_a:
                if a_index in killed_b:
                    yield ('conflicted-a', self.lines_a[a_index])
                else:
                    yield ('new-a', self.lines_a[a_index])
            else:
                yield ('killed-b', self.lines_a[a_index])
        for b_index in range(last_j, j):
            if b_index in new_b:
                if b_index in killed_a:
                    yield ('conflicted-b', self.lines_b[b_index])
                else:
                    yield ('new-b', self.lines_b[b_index])
            else:
                yield ('killed-a', self.lines_b[b_index])
        for a_index in range(i, i + n):
            yield ('unchanged', self.lines_a[a_index])
        last_i = i + n
        last_j = j + n