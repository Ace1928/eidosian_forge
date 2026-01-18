import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
def _sanity_check_key(index_or_builder, key):
    """Raise BadIndexKey if key cannot be used for prefix matching."""
    if key[0] is None:
        raise BadIndexKey(key)
    if len(key) != index_or_builder._key_length:
        raise BadIndexKey(key)