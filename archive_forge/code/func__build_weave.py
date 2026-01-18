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
def _build_weave(self):
    from .bzr import weave
    from .tsort import merge_sort
    self._weave = weave.Weave(weave_name='in_memory_weave', allow_reserved=True)
    parent_map = self._find_recursive_lcas()
    all_texts = self._get_interesting_texts(parent_map)
    tip_key = self._key_prefix + (_mod_revision.CURRENT_REVISION,)
    parent_map[tip_key] = (self.a_key, self.b_key)
    for seq_num, key, depth, eom in reversed(merge_sort(parent_map, tip_key)):
        if key == tip_key:
            continue
        parent_keys = parent_map[key]
        revision_id = key[-1]
        parent_ids = [k[-1] for k in parent_keys]
        self._weave.add_lines(revision_id, parent_ids, all_texts[revision_id])