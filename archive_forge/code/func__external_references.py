import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
def _external_references(self):
    """Return references that are not present in this index.
        """
    keys = set()
    refs = set()
    if self.reference_lists > 1:
        for node in self.iter_all_entries():
            keys.add(node[1])
            refs.update(node[3][1])
        return refs - keys
    else:
        return set()