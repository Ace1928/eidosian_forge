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
class _MPDiffGenerator:
    """Pull out the functionality for generating mp_diffs."""

    def __init__(self, vf, keys):
        self.vf = vf
        self.ordered_keys = tuple(keys)
        self.needed_keys = ()
        self.diffs = {}
        self.parent_map = {}
        self.ghost_parents = ()
        self.refcounts = {}
        self.chunks = {}

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

    def _compute_diff(self, key, parent_lines, lines):
        """Compute a single mp_diff, and store it in self._diffs"""
        if len(parent_lines) > 0:
            left_parent_blocks = self.vf._extract_blocks(key, parent_lines[0], lines)
        else:
            left_parent_blocks = None
        diff = multiparent.MultiParent.from_lines(lines, parent_lines, left_parent_blocks)
        self.diffs[key] = diff

    def _process_one_record(self, key, this_chunks):
        parent_keys = None
        if key in self.parent_map:
            parent_keys = self.parent_map.pop(key)
            if parent_keys is None:
                parent_keys = ()
            parent_lines = []
            for p in parent_keys:
                if p in self.ghost_parents:
                    continue
                refcount = self.refcounts[p]
                if refcount == 1:
                    self.refcounts.pop(p)
                    parent_chunks = self.chunks.pop(p)
                else:
                    self.refcounts[p] = refcount - 1
                    parent_chunks = self.chunks[p]
                p_lines = osutils.chunks_to_lines(parent_chunks)
                parent_lines.append(p_lines)
                del p_lines
            lines = osutils.chunks_to_lines(this_chunks)
            this_chunks = lines
            self._compute_diff(key, parent_lines, lines)
            del lines
        if key in self.refcounts:
            self.chunks[key] = this_chunks

    def _extract_diffs(self):
        needed_keys, refcounts = self._find_needed_keys()
        for record in self.vf.get_record_stream(needed_keys, 'topological', True):
            if record.storage_kind == 'absent':
                raise errors.RevisionNotPresent(record.key, self.vf)
            self._process_one_record(record.key, record.get_bytes_as('chunked'))

    def compute_diffs(self):
        self._extract_diffs()
        dpop = self.diffs.pop
        return [dpop(k) for k in self.ordered_keys]