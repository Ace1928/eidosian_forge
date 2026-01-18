from .. import errors
from ..osutils import basename
from ..revision import NULL_REVISION
from . import inventory
def _reference_content(entry):
    """Serialize the content component of entry which is a tree-reference.

    :param entry: A TreeReference.
    """
    tree_revision = entry.reference_revision
    if tree_revision is None:
        raise InventoryDeltaError('Missing reference revision for %(fileid)r', fileid=entry.file_id)
    return b'tree\x00%s' % tree_revision