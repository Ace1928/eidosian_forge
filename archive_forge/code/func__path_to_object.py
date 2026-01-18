import os
import pygit2
from fsspec.spec import AbstractFileSystem
from .memory import MemoryFile
def _path_to_object(self, path, ref):
    comm, ref = self.repo.resolve_refish(ref or self.ref)
    parts = path.split('/')
    tree = comm.tree
    for part in parts:
        if part and isinstance(tree, pygit2.Tree):
            tree = tree[part]
    return tree