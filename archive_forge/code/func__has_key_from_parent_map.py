import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
def _has_key_from_parent_map(self, key):
    """Check if this index has one key.

    If it's possible to check for multiple keys at once through
    calling get_parent_map that should be faster.
    """
    return key in self.get_parent_map([key])