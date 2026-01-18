from io import BytesIO
from ..lazy_import import lazy_import
import bisect
import math
import tempfile
import zlib
from .. import (chunk_writer, debug, fifo_cache, lru_cache, osutils, trace,
from . import index, static_tuple
from .index import _OPTION_KEY_ELEMENTS, _OPTION_LEN, _OPTION_NODE_REFS
def _get_nodes_by_key(self):
    if self._nodes_by_key is None:
        nodes_by_key = {}
        if self.reference_lists:
            for key, (references, value) in self._nodes.items():
                key_dict = nodes_by_key
                for subkey in key[:-1]:
                    key_dict = key_dict.setdefault(subkey, {})
                key_dict[key[-1]] = (key, value, references)
        else:
            for key, (references, value) in self._nodes.items():
                key_dict = nodes_by_key
                for subkey in key[:-1]:
                    key_dict = key_dict.setdefault(subkey, {})
                key_dict[key[-1]] = (key, value)
        self._nodes_by_key = nodes_by_key
    return self._nodes_by_key