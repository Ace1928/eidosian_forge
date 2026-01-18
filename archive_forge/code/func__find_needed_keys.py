import itertools
import os
import struct
from copy import copy
from io import BytesIO
from typing import Any, Tuple
from zlib import adler32
from ..lazy_import import lazy_import
import fastbencode as bencode
from breezy import (
from breezy.bzr import (
from .. import errors
from .. import graph as _mod_graph
from .. import osutils
from .. import transport as _mod_transport
from ..registry import Registry
from ..textmerge import TextMerge
from . import index
def _find_needed_keys(self):
    """Find the set of keys we need to request.

        This includes all the original keys passed in, and the non-ghost
        parents of those keys.

        :return: (needed_keys, refcounts)
            needed_keys is the set of all texts we need to extract
            refcounts is a dict of {key: num_children} letting us know when we
                no longer need to cache a given parent text
        """
    needed_keys = set(self.ordered_keys)
    parent_map = self.vf.get_parent_map(needed_keys)
    self.parent_map = parent_map
    missing_keys = needed_keys.difference(parent_map)
    if missing_keys:
        raise errors.RevisionNotPresent(list(missing_keys)[0], self.vf)
    refcounts = {}
    setdefault = refcounts.setdefault
    just_parents = set()
    for child_key, parent_keys in parent_map.items():
        if not parent_keys:
            continue
        just_parents.update(parent_keys)
        needed_keys.update(parent_keys)
        for p in parent_keys:
            refcounts[p] = setdefault(p, 0) + 1
    just_parents.difference_update(parent_map)
    self.present_parents = set(self.vf.get_parent_map(just_parents))
    self.ghost_parents = just_parents.difference(self.present_parents)
    needed_keys.difference_update(self.ghost_parents)
    self.needed_keys = needed_keys
    self.refcounts = refcounts
    return (needed_keys, refcounts)