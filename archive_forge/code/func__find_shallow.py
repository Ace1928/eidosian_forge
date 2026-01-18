import collections
import os
import socket
import sys
import time
from functools import partial
from typing import Dict, Iterable, List, Optional, Set, Tuple
import socketserver
import zlib
from dulwich import log_utils
from .archive import tar_stream
from .errors import (
from .object_store import peel_sha
from .objects import Commit, ObjectID, valid_hexsha
from .pack import ObjectContainer, PackedObjectContainer, write_pack_from_container
from .protocol import (
from .refs import PEELED_TAG_SUFFIX, RefsContainer, write_info_refs
from .repo import BaseRepo, Repo
def _find_shallow(store: ObjectContainer, heads, depth):
    """Find shallow commits according to a given depth.

    Args:
      store: An ObjectStore for looking up objects.
      heads: Iterable of head SHAs to start walking from.
      depth: The depth of ancestors to include. A depth of one includes
        only the heads themselves.
    Returns: A tuple of (shallow, not_shallow), sets of SHAs that should be
        considered shallow and unshallow according to the arguments. Note that
        these sets may overlap if a commit is reachable along multiple paths.
    """
    parents: Dict[bytes, List[bytes]] = {}

    def get_parents(sha):
        result = parents.get(sha, None)
        if not result:
            result = store[sha].parents
            parents[sha] = result
        return result
    todo = []
    for head_sha in heads:
        _unpeeled, peeled = peel_sha(store, head_sha)
        if isinstance(peeled, Commit):
            todo.append((peeled.id, 1))
    not_shallow = set()
    shallow = set()
    while todo:
        sha, cur_depth = todo.pop()
        if cur_depth < depth:
            not_shallow.add(sha)
            new_depth = cur_depth + 1
            todo.extend(((p, new_depth) for p in get_parents(sha)))
        else:
            shallow.add(sha)
    return (shallow, not_shallow)