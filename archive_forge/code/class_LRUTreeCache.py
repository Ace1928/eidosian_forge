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
class LRUTreeCache:

    def __init__(self, repository):

        def approx_tree_size(tree):
            return len(tree.root_inventory) * 250
        self.repository = repository
        self._cache = lru_cache.LRUSizeCache(max_size=MAX_TREE_CACHE_SIZE, after_cleanup_size=None, compute_size=approx_tree_size)

    def revision_tree(self, revid):
        try:
            tree = self._cache[revid]
        except KeyError:
            tree = self.repository.revision_tree(revid)
            self.add(tree)
        return tree

    def iter_revision_trees(self, revids):
        trees = {}
        todo = []
        for revid in revids:
            try:
                tree = self._cache[revid]
            except KeyError:
                todo.append(revid)
            else:
                if tree.get_revision_id() != revid:
                    raise AssertionError('revision id did not match: {} != {}'.format(tree.get_revision_id(), revid))
                trees[revid] = tree
        for tree in self.repository.revision_trees(todo):
            trees[tree.get_revision_id()] = tree
            self.add(tree)
        return (trees[r] for r in revids)

    def revision_trees(self, revids):
        return list(self.iter_revision_trees(revids))

    def add(self, tree):
        self._cache[tree.get_revision_id()] = tree