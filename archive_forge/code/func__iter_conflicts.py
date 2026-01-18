from io import BytesIO
from ... import conflicts as _mod_conflicts
from ... import errors, lock, osutils
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ...bzr import conflicts as _mod_bzr_conflicts
from ...bzr import inventory
from ...bzr import transform as bzr_transform
from ...bzr import xml5
from ...bzr.workingtree_3 import PreDirStateWorkingTree
from ...mutabletree import MutableTree
from ...transport.local import LocalTransport
from ...workingtree import WorkingTreeFormat
def _iter_conflicts(self):
    conflicted = set()
    for path, file_class, file_kind, entry in self.list_files():
        stem = get_conflicted_stem(path)
        if stem is None:
            continue
        if stem not in conflicted:
            conflicted.add(stem)
            yield stem