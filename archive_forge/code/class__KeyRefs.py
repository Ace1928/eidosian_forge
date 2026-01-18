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
class _KeyRefs:

    def __init__(self, track_new_keys=False):
        self.refs = {}
        if track_new_keys:
            self.new_keys = set()
        else:
            self.new_keys = None

    def clear(self):
        if self.refs:
            self.refs.clear()
        if self.new_keys:
            self.new_keys.clear()

    def add_references(self, key, refs):
        for referenced in refs:
            try:
                needed_by = self.refs[referenced]
            except KeyError:
                needed_by = self.refs[referenced] = set()
            needed_by.add(key)
        self.add_key(key)

    def get_new_keys(self):
        return self.new_keys

    def get_unsatisfied_refs(self):
        return self.refs.keys()

    def _satisfy_refs_for_key(self, key):
        try:
            del self.refs[key]
        except KeyError:
            pass

    def add_key(self, key):
        self._satisfy_refs_for_key(key)
        if self.new_keys is not None:
            self.new_keys.add(key)

    def satisfy_refs_for_keys(self, keys):
        for key in keys:
            self._satisfy_refs_for_key(key)

    def get_referrers(self):
        return set(itertools.chain.from_iterable(self.refs.values()))