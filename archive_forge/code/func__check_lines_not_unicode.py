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
def _check_lines_not_unicode(self, lines):
    """Check that lines being added to a versioned file are not unicode."""
    for line in lines:
        if line.__class__ is not bytes:
            raise errors.BzrBadParameterUnicode('lines')