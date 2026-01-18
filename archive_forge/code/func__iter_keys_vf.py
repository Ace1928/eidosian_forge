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
def _iter_keys_vf(self, keys):
    prefixes = self._partition_keys(keys)
    sha1s = {}
    for prefix, suffixes in prefixes.items():
        path = self._mapper.map(prefix)
        vf = self._get_vf(path)
        yield (prefix, suffixes, vf)