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
def _get_base_is_other_ancestor(self):
    if self._base_is_other_ancestor is None:
        if self.other_basis is None:
            return True
        self._base_is_other_ancestor = self.revision_graph.is_ancestor(self.base_rev_id, self.other_basis)
    return self._base_is_other_ancestor