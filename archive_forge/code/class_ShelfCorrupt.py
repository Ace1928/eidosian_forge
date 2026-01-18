import errno
import re
import fastbencode as bencode
from . import errors
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
class ShelfCorrupt(errors.BzrError):
    _fmt = 'Shelf corrupt.'