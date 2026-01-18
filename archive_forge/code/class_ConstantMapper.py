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
class ConstantMapper(KeyMapper):
    """A key mapper that maps to a constant result."""

    def __init__(self, result):
        """Create a ConstantMapper which will return result for all maps."""
        self._result = result

    def map(self, key):
        """See KeyMapper.map()."""
        return self._result