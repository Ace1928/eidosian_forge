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
def _generate_inventory_delta(self):
    """Generate an inventory delta for the current transform."""
    inventory_delta = []
    new_paths = self._inventory_altered()
    total_entries = len(new_paths) + len(self._removed_id)
    with ui.ui_factory.nested_progress_bar() as child_pb:
        for num, trans_id in enumerate(self._removed_id):
            if num % 10 == 0:
                child_pb.update(gettext('removing file'), num, total_entries)
            if trans_id == self._new_root:
                file_id = self._tree.path2id('')
            else:
                file_id = self.tree_file_id(trans_id)
            if file_id in self._r_new_id:
                continue
            path = self._tree_id_paths[trans_id]
            inventory_delta.append((path, None, file_id, None))
        new_path_file_ids = {t: self.final_file_id(t) for p, t in new_paths}
        for num, (path, trans_id) in enumerate(new_paths):
            if num % 10 == 0:
                child_pb.update(gettext('adding file'), num + len(self._removed_id), total_entries)
            file_id = new_path_file_ids[trans_id]
            if file_id is None:
                continue
            kind = self.final_kind(trans_id)
            if kind is None:
                kind = self._tree.stored_kind(self._tree.id2path(file_id))
            parent_trans_id = self.final_parent(trans_id)
            parent_file_id = new_path_file_ids.get(parent_trans_id)
            if parent_file_id is None:
                parent_file_id = self.final_file_id(parent_trans_id)
            if trans_id in self._new_reference_revision:
                new_entry = inventory.TreeReference(file_id, self._new_name[trans_id], self.final_file_id(self._new_parent[trans_id]), None, self._new_reference_revision[trans_id])
            else:
                new_entry = inventory.make_entry(kind, self.final_name(trans_id), parent_file_id, file_id)
            try:
                old_path = self._tree.id2path(new_entry.file_id)
            except errors.NoSuchId:
                old_path = None
            new_executability = self._new_executability.get(trans_id)
            if new_executability is not None:
                new_entry.executable = new_executability
            inventory_delta.append((old_path, path, new_entry.file_id, new_entry))
    return inventory_delta