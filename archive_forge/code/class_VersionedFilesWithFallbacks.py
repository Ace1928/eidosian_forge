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
class VersionedFilesWithFallbacks(VersionedFiles):

    def without_fallbacks(self):
        """Return a clone of this object without any fallbacks configured."""
        raise NotImplementedError(self.without_fallbacks)

    def add_fallback_versioned_files(self, a_versioned_files):
        """Add a source of texts for texts not present in this knit.

        :param a_versioned_files: A VersionedFiles object.
        """
        raise NotImplementedError(self.add_fallback_versioned_files)

    def get_known_graph_ancestry(self, keys):
        """Get a KnownGraph instance with the ancestry of keys."""
        parent_map, missing_keys = self._index.find_ancestry(keys)
        for fallback in self._transitive_fallbacks():
            if not missing_keys:
                break
            f_parent_map, f_missing_keys = fallback._index.find_ancestry(missing_keys)
            parent_map.update(f_parent_map)
            missing_keys = f_missing_keys
        kg = _mod_graph.KnownGraph(parent_map)
        return kg