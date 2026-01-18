import collections
import heapq
from itertools import chain
from typing import Deque, Dict, List, Optional, Set, Tuple
from .diff_tree import (
from .errors import MissingCommitError
from .objects import Commit, ObjectID, Tag
class WalkEntry:
    """Object encapsulating a single result from a walk."""

    def __init__(self, walker, commit) -> None:
        self.commit = commit
        self._store = walker.store
        self._get_parents = walker.get_parents
        self._changes: Dict[str, List[TreeChange]] = {}
        self._rename_detector = walker.rename_detector

    def changes(self, path_prefix=None):
        """Get the tree changes for this entry.

        Args:
          path_prefix: Portion of the path in the repository to
            use to filter changes. Must be a directory name. Must be
            a full, valid, path reference (no partial names or wildcards).
        Returns: For commits with up to one parent, a list of TreeChange
            objects; if the commit has no parents, these will be relative to
            the empty tree. For merge commits, a list of lists of TreeChange
            objects; see dulwich.diff.tree_changes_for_merge.
        """
        cached = self._changes.get(path_prefix)
        if cached is None:
            commit = self.commit
            if not self._get_parents(commit):
                changes_func = tree_changes
                parent = None
            elif len(self._get_parents(commit)) == 1:
                changes_func = tree_changes
                parent = self._store[self._get_parents(commit)[0]].tree
                if path_prefix:
                    mode, subtree_sha = parent.lookup_path(self._store.__getitem__, path_prefix)
                    parent = self._store[subtree_sha]
            else:
                changes_func = tree_changes_for_merge
                parent = [self._store[p].tree for p in self._get_parents(commit)]
                if path_prefix:
                    parent_trees = [self._store[p] for p in parent]
                    parent = []
                    for p in parent_trees:
                        try:
                            mode, st = p.lookup_path(self._store.__getitem__, path_prefix)
                        except KeyError:
                            pass
                        else:
                            parent.append(st)
            commit_tree_sha = commit.tree
            if path_prefix:
                commit_tree = self._store[commit_tree_sha]
                mode, commit_tree_sha = commit_tree.lookup_path(self._store.__getitem__, path_prefix)
            cached = list(changes_func(self._store, parent, commit_tree_sha, rename_detector=self._rename_detector))
            self._changes[path_prefix] = cached
        return self._changes[path_prefix]

    def __repr__(self) -> str:
        return f'<WalkEntry commit={self.commit.id}, changes={self.changes()!r}>'