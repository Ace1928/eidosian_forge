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
def _default_other_winner_merge(self, merge_hook_params):
    """Replace this contents with other."""
    trans_id = merge_hook_params.trans_id
    if merge_hook_params.other_path is not None:
        transform.create_from_tree(self.tt, trans_id, self.other_tree, merge_hook_params.other_path, filter_tree_path=self._get_filter_tree_path(merge_hook_params.other_path))
        return ('done', None)
    elif merge_hook_params.this_path is not None:
        return ('delete', None)
    else:
        raise AssertionError('winner is OTHER, but file %r not in THIS or OTHER tree' % (merge_hook_params.base_path,))