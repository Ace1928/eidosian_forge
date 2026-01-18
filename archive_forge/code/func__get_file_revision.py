import contextlib
import errno
import os
import tempfile
import time
from stat import S_IEXEC, S_ISREG
from .. import (annotate, conflicts, controldir, errors, lock, multiparent,
from .. import revision as _mod_revision
from .. import trace
from .. import transport as _mod_transport
from .. import tree, ui, urlutils
from ..filters import ContentFilterContext, filtered_output_bytes
from ..i18n import gettext
from ..mutabletree import MutableTree
from ..progress import ProgressPhase
from ..transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ..tree import find_previous_path
from . import inventory, inventorytree
from .conflicts import Conflict
def _get_file_revision(self, path, file_id, vf, tree_revision):
    parent_keys = [(file_id, t.get_file_revision(t.id2path(file_id))) for t in self._iter_parent_trees()]
    vf.add_lines((file_id, tree_revision), parent_keys, self.get_file_lines(path))
    repo = self._get_repository()
    base_vf = repo.texts
    if base_vf not in vf.fallback_versionedfiles:
        vf.fallback_versionedfiles.append(base_vf)
    return tree_revision