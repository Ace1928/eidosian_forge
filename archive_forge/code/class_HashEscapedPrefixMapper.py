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
class HashEscapedPrefixMapper(HashPrefixMapper):
    """Combines the escaped first component of a key with a hash.

    This mapper is for use with a transport based backend.
    """
    _safe = bytearray(b'abcdefghijklmnopqrstuvwxyz0123456789-_@,.')

    def _escape(self, prefix):
        """Turn a key element into a filesystem safe string.

        This is similar to a plain urlutils.quote, except
        it uses specific safe characters, so that it doesn't
        have to translate a lot of valid file ids.
        """
        r = [c in self._safe and chr(c) or '%%%02x' % c for c in bytearray(prefix)]
        return ''.join(r).encode('ascii')

    def _unescape(self, basename):
        """Escaped names are easily unescaped by urlutils."""
        return urlutils.unquote(basename)