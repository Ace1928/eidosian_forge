from typing import FrozenSet, Optional, Set, Tuple
import gevent
from gevent import pool
from .object_store import (
from .objects import Commit, ObjectID, Tag
class GreenThreadsMissingObjectFinder(MissingObjectFinder):
    """Find the objects missing from another object store.

    Same implementation as object_store.MissingObjectFinder
    except we use gevent to parallelize object retrieval.
    """

    def __init__(self, object_store, haves, wants, progress=None, get_tagged=None, concurrency=1, get_parents=None) -> None:

        def collect_tree_sha(sha):
            self.sha_done.add(sha)
            cmt = object_store[sha]
            _collect_filetree_revs(object_store, cmt.tree, self.sha_done)
        self.object_store = object_store
        p = pool.Pool(size=concurrency)
        have_commits, have_tags = _split_commits_and_tags(object_store, haves, ignore_unknown=True, pool=p)
        want_commits, want_tags = _split_commits_and_tags(object_store, wants, ignore_unknown=False, pool=p)
        all_ancestors: FrozenSet[ObjectID] = frozenset(_collect_ancestors(object_store, have_commits)[0])
        missing_commits, common_commits = _collect_ancestors(object_store, want_commits, all_ancestors)
        self.sha_done = set()
        jobs = [p.spawn(collect_tree_sha, c) for c in common_commits]
        gevent.joinall(jobs)
        for t in have_tags:
            self.sha_done.add(t)
        missing_tags = want_tags.difference(have_tags)
        wants = missing_commits.union(missing_tags)
        self.objects_to_send: Set[Tuple[ObjectID, Optional[bytes], Optional[int], bool]] = {(w, None, 0, False) for w in wants}
        if progress is None:
            self.progress = lambda x: None
        else:
            self.progress = progress
        self._tagged = get_tagged and get_tagged() or {}