from .. import errors
from ..osutils import basename
from ..revision import NULL_REVISION
from . import inventory
def _link_content(entry):
    """Serialize the content component of entry which is a symlink.

    :param entry: An InventoryLink.
    """
    target = entry.symlink_target
    if target is None:
        raise InventoryDeltaError('Missing target for %(fileid)r', fileid=entry.file_id)
    return b'link\x00%s' % target.encode('utf8')