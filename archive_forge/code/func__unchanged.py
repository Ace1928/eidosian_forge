from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def _unchanged(self, previous_ie):
    """See InventoryEntry._unchanged."""
    compatible = super()._unchanged(previous_ie)
    if self.reference_revision != previous_ie.reference_revision:
        compatible = False
    return compatible