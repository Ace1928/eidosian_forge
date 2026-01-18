import contextlib
import errno
import os
import sys
from typing import TYPE_CHECKING, Optional, Tuple
import breezy
from .lazy_import import lazy_import
import stat
from breezy import (
from . import errors, mutabletree, osutils
from . import revision as _mod_revision
from .controldir import (ControlComponent, ControlComponentFormat,
from .i18n import gettext
from .symbol_versioning import deprecated_in, deprecated_method
from .trace import mutter, note
from .transport import NoSuchFile
def _filter_parent_ids_by_ancestry(self, revision_ids):
    """Check that all merged revisions are proper 'heads'.

        This will always return the first revision_id, and any merged revisions
        which are
        """
    if len(revision_ids) == 0:
        return revision_ids
    graph = self.branch.repository.get_graph()
    heads = graph.heads(revision_ids)
    new_revision_ids = revision_ids[:1]
    for revision_id in revision_ids[1:]:
        if revision_id in heads and revision_id not in new_revision_ids:
            new_revision_ids.append(revision_id)
    if new_revision_ids != revision_ids:
        mutter('requested to set revision_ids = %s, but filtered to %s', revision_ids, new_revision_ids)
    return new_revision_ids