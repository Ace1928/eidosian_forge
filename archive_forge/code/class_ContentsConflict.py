import errno
import itertools
import os
import posixpath
import re
import stat
import sys
from collections import defaultdict
from dulwich.config import ConfigFile as GitConfigFile
from dulwich.file import FileLocked, GitFile
from dulwich.ignore import IgnoreFilterManager
from dulwich.index import (ConflictedIndexEntry, Index, IndexEntry, SHA1Writer,
from dulwich.object_store import iter_tree_contents
from dulwich.objects import S_ISGITLINK
from .. import branch as _mod_branch
from .. import conflicts as _mod_conflicts
from .. import controldir as _mod_controldir
from .. import errors, globbing, lock, osutils
from .. import revision as _mod_revision
from .. import trace
from .. import transport as _mod_transport
from .. import tree, urlutils, workingtree
from ..decorators import only_raises
from ..mutabletree import BadReferenceTarget, MutableTree
from .dir import BareLocalGitControlDirFormat, LocalGitDir
from .mapping import decode_git_path, encode_git_path, mode_kind
from .tree import MutableGitIndexTree
class ContentsConflict(_mod_conflicts.Conflict):
    """The files are of different types (or both binary), or not present."""
    has_files = True
    typestring = 'contents conflict'
    format = 'Contents conflict in %(path)s'

    def __init__(self, path, conflict_path=None):
        for suffix in ('.BASE', '.THIS', '.OTHER'):
            if path.endswith(suffix):
                path = path[:-len(suffix)]
                break
        _mod_conflicts.Conflict.__init__(self, path)
        self.conflict_path = conflict_path

    def _revision_tree(self, tree, revid):
        return tree.branch.repository.revision_tree(revid)

    def associated_filenames(self):
        return [self.path + suffix for suffix in ('.BASE', '.OTHER', '.THIS')]

    def _resolve(self, tt, suffix_to_remove):
        """Resolve the conflict.

        :param tt: The TreeTransform where the conflict is resolved.
        :param suffix_to_remove: Either 'THIS' or 'OTHER'

        The resolution is symmetric: when taking THIS, OTHER is deleted and
        item.THIS is renamed into item and vice-versa.
        """
        try:
            tt.delete_contents(tt.trans_id_tree_path(self.path + '.' + suffix_to_remove))
        except _mod_transport.NoSuchFile:
            pass
        try:
            this_path = tt._tree.id2path(self.file_id)
        except errors.NoSuchId:
            this_tid = None
        else:
            this_tid = tt.trans_id_tree_path(this_path)
        if this_tid is not None:
            parent_tid = tt.get_tree_parent(this_tid)
            tt.adjust_path(osutils.basename(self.path), parent_tid, this_tid)
            tt.apply()

    def _resolve_with_cleanups(self, tree, *args, **kwargs):
        with tree.transform() as tt:
            self._resolve(tt, *args, **kwargs)

    def action_take_this(self, tree):
        self._resolve_with_cleanups(tree, 'OTHER')

    def action_take_other(self, tree):
        self._resolve_with_cleanups(tree, 'THIS')

    @classmethod
    def from_index_entry(cls, entry):
        """Create a conflict from a Git index entry."""
        return cls(entry.path)

    def describe(self):
        return f'Contents conflict in {self.__dict__['path']}'

    def to_index_entry(self, tree):
        """Convert the conflict to a Git index entry."""
        encoded_path = encode_git_path(tree.abspath(self.path))
        try:
            base = index_entry_from_path(encoded_path + b'.BASE')
        except FileNotFoundError:
            base = None
        try:
            other = index_entry_from_path(encoded_path + b'.OTHER')
        except FileNotFoundError:
            other = None
        try:
            this = index_entry_from_path(encoded_path + b'.THIS')
        except FileNotFoundError:
            this = None
        return ConflictedIndexEntry(this=this, other=other, ancestor=base)