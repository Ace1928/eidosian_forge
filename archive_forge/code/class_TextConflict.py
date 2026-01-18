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
class TextConflict(_mod_conflicts.Conflict):
    """The merge algorithm could not resolve all differences encountered."""
    has_files = True
    typestring = 'text conflict'
    _conflict_re = re.compile(b'^(<{7}|={7}|>{7})')

    def __init__(self, path):
        super(TextConflict, self).__init__(path)

    def associated_filenames(self):
        return [self.path + suffix for suffix in ('.BASE', '.OTHER', '.THIS')]

    def _resolve(self, tt, winner_suffix):
        """Resolve the conflict by copying one of .THIS or .OTHER into file.

        :param tt: The TreeTransform where the conflict is resolved.
        :param winner_suffix: Either 'THIS' or 'OTHER'

        The resolution is symmetric, when taking THIS, item.THIS is renamed
        into item and vice-versa. This takes one of the files as a whole
        ignoring every difference that could have been merged cleanly.
        """
        item_tid = tt.trans_id_tree_path(self.path)
        item_parent_tid = tt.get_tree_parent(item_tid)
        winner_path = self.path + '.' + winner_suffix
        winner_tid = tt.trans_id_tree_path(winner_path)
        winner_parent_tid = tt.get_tree_parent(winner_tid)
        tt.adjust_path(osutils.basename(self.path), winner_parent_tid, winner_tid)
        tt.adjust_path(osutils.basename(winner_path), item_parent_tid, item_tid)
        tt.unversion_file(item_tid)
        tt.version_file(winner_tid)
        tt.apply()

    def action_auto(self, tree):
        try:
            kind = tree.kind(self.path)
        except _mod_transport.NoSuchFile:
            return
        if kind != 'file':
            raise NotImplementedError('Conflict is not a file')
        conflict_markers_in_line = self._conflict_re.search
        with tree.get_file(self.path) as f:
            for line in f:
                if conflict_markers_in_line(line):
                    raise NotImplementedError('Conflict markers present')

    def _resolve_with_cleanups(self, tree, *args, **kwargs):
        with tree.transform() as tt:
            self._resolve(tt, *args, **kwargs)

    def action_take_this(self, tree):
        self._resolve_with_cleanups(tree, 'THIS')

    def action_take_other(self, tree):
        self._resolve_with_cleanups(tree, 'OTHER')

    def do(self, action, tree):
        """Apply the specified action to the conflict.

        :param action: The method name to call.

        :param tree: The tree passed as a parameter to the method.
        """
        meth = getattr(self, 'action_%s' % action, None)
        if meth is None:
            raise NotImplementedError(self.__class__.__name__ + '.' + action)
        meth(tree)

    def action_done(self, tree):
        """Mark the conflict as solved once it has been handled."""
        pass

    def describe(self):
        return 'Text conflict in %(path)s' % self.__dict__

    def __str__(self):
        return self.describe()

    def __repr__(self):
        return '{}({!r})'.format(type(self).__name__, self.path)

    @classmethod
    def from_index_entry(cls, path, entry):
        """Create a conflict from a Git index entry."""
        return cls(path)

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