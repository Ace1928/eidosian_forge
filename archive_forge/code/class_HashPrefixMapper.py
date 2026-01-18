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
class HashPrefixMapper(URLEscapeMapper):
    """A key mapper that combines the first component of a key with a hash.

    This mapper is for use with a transport based backend.
    """

    def _map(self, key):
        """See KeyMapper.map()."""
        prefix = self._escape(key[0])
        return '{:02x}/{}'.format(adler32(prefix) & 255, prefix.decode('utf-8'))

    def _escape(self, prefix):
        """No escaping needed here."""
        return prefix

    def _unmap(self, partition_id):
        """See KeyMapper.unmap()."""
        return (self._unescape(osutils.basename(partition_id)).encode('utf-8'),)

    def _unescape(self, basename):
        """No unescaping needed for HashPrefixMapper."""
        return basename