from .. import errors
from ..osutils import basename
from ..revision import NULL_REVISION
from . import inventory
def _dir_to_entry(content, name, parent_id, file_id, last_modified, _type=inventory.InventoryDirectory):
    """Convert a dir content record to an InventoryDirectory."""
    result = _type(file_id, name, parent_id)
    result.revision = last_modified
    return result