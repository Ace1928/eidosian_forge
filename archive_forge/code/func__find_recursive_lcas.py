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
def _find_recursive_lcas(self):
    """Find all the ancestors back to a unique lca"""
    cur_ancestors = (self.a_key, self.b_key)
    parent_map = {}
    while True:
        next_lcas = self.graph.find_lca(*cur_ancestors)
        if next_lcas == {_mod_revision.NULL_REVISION}:
            next_lcas = ()
        for rev_key in cur_ancestors:
            ordered_parents = tuple(self.graph.find_merge_order(rev_key, next_lcas))
            parent_map[rev_key] = ordered_parents
        if len(next_lcas) == 0:
            break
        elif len(next_lcas) == 1:
            parent_map[list(next_lcas)[0]] = ()
            break
        elif len(next_lcas) > 2:
            trace.mutter('More than 2 LCAs, falling back to all nodes for: %s, %s\n=> %s', self.a_key, self.b_key, cur_ancestors)
            cur_lcas = next_lcas
            while len(cur_lcas) > 1:
                cur_lcas = self.graph.find_lca(*cur_lcas)
            if len(cur_lcas) == 0:
                unique_lca = None
            else:
                unique_lca = list(cur_lcas)[0]
                if unique_lca == _mod_revision.NULL_REVISION:
                    unique_lca = None
            parent_map.update(self._find_unique_parents(next_lcas, unique_lca))
            break
        cur_ancestors = next_lcas
    return parent_map