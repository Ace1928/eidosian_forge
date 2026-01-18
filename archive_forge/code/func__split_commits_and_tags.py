from typing import FrozenSet, Optional, Set, Tuple
import gevent
from gevent import pool
from .object_store import (
from .objects import Commit, ObjectID, Tag
def _split_commits_and_tags(obj_store, lst, *, ignore_unknown=False, pool=None):
    """Split object id list into two list with commit SHA1s and tag SHA1s.

    Same implementation as object_store._split_commits_and_tags
    except we use gevent to parallelize object retrieval.
    """
    commits = set()
    tags = set()

    def find_commit_type(sha):
        try:
            o = obj_store[sha]
        except KeyError:
            if not ignore_unknown:
                raise
        else:
            if isinstance(o, Commit):
                commits.add(sha)
            elif isinstance(o, Tag):
                tags.add(sha)
                commits.add(o.object[1])
            else:
                raise KeyError('Not a commit or a tag: %s' % sha)
    jobs = [pool.spawn(find_commit_type, s) for s in lst]
    gevent.joinall(jobs)
    return (commits, tags)