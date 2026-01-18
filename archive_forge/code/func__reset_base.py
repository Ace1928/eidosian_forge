import stat
from typing import Dict, Tuple
from fastimport import commands, parser, processor
from fastimport import errors as fastimport_errors
from .index import commit_tree
from .object_store import iter_tree_contents
from .objects import ZERO_SHA, Blob, Commit, Tag
def _reset_base(self, commit_id):
    if self.last_commit == commit_id:
        return
    self._contents = {}
    self.last_commit = commit_id
    if commit_id != ZERO_SHA:
        tree_id = self.repo[commit_id].tree
        for path, mode, hexsha in iter_tree_contents(self.repo.object_store, tree_id):
            self._contents[path] = (mode, hexsha)