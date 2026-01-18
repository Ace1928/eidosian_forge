from io import BytesIO
from dulwich.errors import NotCommitError
from dulwich.object_store import peel_sha, tree_lookup_path
from dulwich.objects import ZERO_SHA, Commit
from .. import check, errors
from .. import graph as _mod_graph
from .. import lock, repository
from .. import revision as _mod_revision
from .. import trace, transactions, ui
from ..decorators import only_raises
from ..foreign import ForeignRepository
from .filegraph import GitFileLastChangeScanner, GitFileParentProvider
from .mapping import (default_mapping, encode_git_path, foreign_vcs_git,
from .tree import GitRevisionTree
def _get_parent_map_no_fallbacks(self, revids):
    return self.get_parent_map(revids, no_alternates=True)