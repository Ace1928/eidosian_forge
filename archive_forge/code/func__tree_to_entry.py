from .. import errors
from ..osutils import basename
from ..revision import NULL_REVISION
from . import inventory
def _tree_to_entry(content, name, parent_id, file_id, last_modified, _type=inventory.TreeReference):
    """Convert a tree content record to a TreeReference."""
    result = _type(file_id, name, parent_id)
    result.revision = last_modified
    result.reference_revision = content[1]
    return result