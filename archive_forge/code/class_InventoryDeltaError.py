from .. import errors
from ..osutils import basename
from ..revision import NULL_REVISION
from . import inventory
class InventoryDeltaError(errors.BzrError):
    """An error when serializing or deserializing an inventory delta."""
    internal_error = True

    def __init__(self, format_string, **kwargs):
        self._fmt = format_string
        super().__init__(**kwargs)