import posixpath
import stat
from typing import Dict, Iterable, Iterator, List
from dulwich.object_store import BaseObjectStore
from dulwich.objects import (ZERO_SHA, Blob, Commit, ObjectID, ShaFile, Tree,
from dulwich.pack import Pack, PackData, pack_objects_to_data
from .. import errors, lru_cache, osutils, trace, ui
from ..bzr.testament import StrictTestament3
from ..lock import LogicalLockResult
from ..revision import NULL_REVISION
from ..tree import InterTree
from .cache import from_repository as cache_from_repository
from .mapping import (default_mapping, encode_git_path, entry_mode,
from .unpeel_map import UnpeelMap
def _reconstruct_commit(self, rev, tree_sha, lossy, verifiers):
    """Reconstruct a Commit object.

        :param rev: Revision object
        :param tree_sha: SHA1 of the root tree object
        :param lossy: Whether or not to roundtrip bzr metadata
        :param verifiers: Verifiers for the commits
        :return: Commit object
        """

    def parent_lookup(revid):
        try:
            return self._lookup_revision_sha1(revid)
        except errors.NoSuchRevision:
            return None
    return self.mapping.export_commit(rev, tree_sha, parent_lookup, lossy, verifiers)