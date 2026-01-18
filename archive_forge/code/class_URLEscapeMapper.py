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
class URLEscapeMapper(KeyMapper):
    """Base class for use with transport backed storage.

    This provides a map and unmap wrapper that respectively url escape and
    unescape their outputs and inputs.
    """

    def map(self, key):
        """See KeyMapper.map()."""
        return urlutils.quote(self._map(key))

    def unmap(self, partition_id):
        """See KeyMapper.unmap()."""
        return self._unmap(urlutils.unquote(partition_id))