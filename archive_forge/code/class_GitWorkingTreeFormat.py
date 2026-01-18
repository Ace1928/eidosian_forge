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
class GitWorkingTreeFormat(workingtree.WorkingTreeFormat):
    _tree_class = GitWorkingTree
    supports_versioned_directories = False
    supports_setting_file_ids = False
    supports_store_uncommitted = False
    supports_leftmost_parent_id_as_ghost = False
    supports_righthand_parent_id_as_ghost = False
    requires_normalized_unicode_filenames = True
    supports_merge_modified = False
    ignore_filename = '.gitignore'

    @property
    def _matchingcontroldir(self):
        from .dir import LocalGitControlDirFormat
        return LocalGitControlDirFormat()

    def get_format_description(self):
        return 'Git Working Tree'

    def initialize(self, a_controldir, revision_id=None, from_branch=None, accelerator_tree=None, hardlink=False):
        """See WorkingTreeFormat.initialize()."""
        if not isinstance(a_controldir, LocalGitDir):
            raise errors.IncompatibleFormat(self, a_controldir)
        branch = a_controldir.open_branch(nascent_ok=True)
        if revision_id is not None:
            branch.set_last_revision(revision_id)
        wt = GitWorkingTree(a_controldir, a_controldir.open_repository(), branch)
        for hook in MutableTree.hooks['post_build_tree']:
            hook(wt)
        return wt