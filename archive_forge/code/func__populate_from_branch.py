import os
import posixpath
import stat
from dulwich.index import index_entry_from_stat
from dulwich.objects import Blob, Tree
from breezy import errors, lock, osutils
from breezy import revision as _mod_revision
from breezy import tree as _mod_tree
from breezy import urlutils
from breezy.transport.memory import MemoryTransport
from .mapping import decode_git_path, encode_git_path
from .tree import MutableGitIndexTree
def _populate_from_branch(self):
    """Populate the in-tree state from the branch."""
    if self.branch.head is None:
        self._parent_ids = []
    else:
        self._parent_ids = [self.last_revision()]
    self._file_transport = MemoryTransport()
    if self.branch.head is None:
        tree = Tree()
    else:
        tree_id = self.store[self.branch.head].tree
        tree = self.store[tree_id]
    trees = [('', tree)]
    while trees:
        path, tree = trees.pop()
        for name, mode, sha in tree.iteritems():
            subpath = posixpath.join(path, decode_git_path(name))
            if stat.S_ISDIR(mode):
                self._file_transport.mkdir(subpath)
                trees.append((subpath, self.store[sha]))
            elif stat.S_ISREG(mode):
                self._file_transport.put_bytes(subpath, self.store[sha].data)
                self._index_add_entry(subpath, 'file')
            else:
                raise NotImplementedError(self._populate_from_branch)