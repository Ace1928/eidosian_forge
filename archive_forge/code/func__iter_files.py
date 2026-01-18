import stat
from typing import Dict, Tuple
from fastimport import commands, parser, processor
from fastimport import errors as fastimport_errors
from .index import commit_tree
from .object_store import iter_tree_contents
from .objects import ZERO_SHA, Blob, Commit, Tag
def _iter_files(self, base_tree, new_tree):
    for (old_path, new_path), (old_mode, new_mode), (old_hexsha, new_hexsha) in self.store.tree_changes(base_tree, new_tree):
        if new_path is None:
            yield commands.FileDeleteCommand(old_path)
            continue
        if not stat.S_ISDIR(new_mode):
            blob = self.store[new_hexsha]
            marker = self.emit_blob(blob)
        if old_path != new_path and old_path is not None:
            yield commands.FileRenameCommand(old_path, new_path)
        if old_mode != new_mode or old_hexsha != new_hexsha:
            prefixed_marker = b':' + marker
            yield commands.FileModifyCommand(new_path, new_mode, prefixed_marker, None)