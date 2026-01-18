import os
from io import BytesIO
from ..lazy_import import lazy_import
import contextlib
import errno
import stat
from breezy import (
from breezy.bzr import (
from .. import errors
from .. import revision as _mod_revision
from ..lock import LogicalLockResult
from ..lockable_files import LockableFiles
from ..lockdir import LockDir
from ..mutabletree import BadReferenceTarget, MutableTree
from ..osutils import file_kind, isdir, pathjoin, realpath, safe_unicode
from ..transport import NoSuchFile, get_transport_from_path
from ..transport.local import LocalTransport
from ..tree import FileTimestampUnavailable, InterTree, MissingNestedTree
from ..workingtree import WorkingTree
from . import dirstate
from .inventory import ROOT_ID, Inventory, entry_factory
from .inventorytree import (InterInventoryTree, InventoryRevisionTree,
from .workingtree import InventoryWorkingTree, WorkingTreeFormatMetaDir
def _generate_inventory(self):
    """Create and set self.inventory from the dirstate object.

        (So this is only called the first time the inventory is requested for
        this tree; it then remains in memory until it's out of date.)

        This is relatively expensive: we have to walk the entire dirstate.
        """
    if not self._locked:
        raise AssertionError('cannot generate inventory of an unlocked dirstate revision tree')
    self._dirstate._read_dirblocks_if_needed()
    if self._revision_id not in self._dirstate.get_parent_ids():
        raise AssertionError('parent {} has disappeared from {}'.format(self._revision_id, self._dirstate.get_parent_ids()))
    parent_index = self._dirstate.get_parent_ids().index(self._revision_id) + 1
    root_key, current_entry = self._dirstate._get_entry(parent_index, path_utf8=b'')
    current_id = root_key[2]
    if current_entry[parent_index][0] != b'd':
        raise AssertionError()
    inv = Inventory(root_id=current_id, revision_id=self._revision_id)
    inv.root.revision = current_entry[parent_index][4]
    minikind_to_kind = dirstate.DirState._minikind_to_kind
    factory = entry_factory
    utf8_decode = cache_utf8._utf8_decode
    inv_byid = inv._byid
    parent_ies = {b'': inv.root}
    for block in self._dirstate._dirblocks[1:]:
        dirname = block[0]
        try:
            parent_ie = parent_ies[dirname]
        except KeyError:
            continue
        for key, entry in block[1]:
            minikind, fingerprint, size, executable, revid = entry[parent_index]
            if minikind in (b'a', b'r'):
                continue
            name = key[1]
            name_unicode = utf8_decode(name)[0]
            file_id = key[2]
            kind = minikind_to_kind[minikind]
            inv_entry = factory[kind](file_id, name_unicode, parent_ie.file_id)
            inv_entry.revision = revid
            if kind == 'file':
                inv_entry.executable = executable
                inv_entry.text_size = size
                inv_entry.text_sha1 = fingerprint
            elif kind == 'directory':
                parent_ies[(dirname + b'/' + name).strip(b'/')] = inv_entry
            elif kind == 'symlink':
                inv_entry.symlink_target = utf8_decode(fingerprint)[0]
            elif kind == 'tree-reference':
                inv_entry.reference_revision = fingerprint or None
            else:
                raise AssertionError('cannot convert entry %r into an InventoryEntry' % entry)
            if file_id in inv_byid:
                raise AssertionError('file_id %s already in inventory as %s' % (file_id, inv_byid[file_id]))
            if name_unicode in parent_ie.children:
                raise AssertionError('name %r already in parent' % (name_unicode,))
            inv_byid[file_id] = inv_entry
            parent_ie.children[name_unicode] = inv_entry
    self._inventory = inv