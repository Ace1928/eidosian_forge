from .. import errors, osutils
from . import inventory, xml6
from .xml_serializer import (encode_and_escape, get_utf8_or_ascii,
def _append_inventory_root(self, append, inv):
    """Append the inventory root to output."""
    if inv.root.file_id not in (None, inventory.ROOT_ID):
        fileid = b''.join([b' file_id="', encode_and_escape(inv.root.file_id), b'"'])
    else:
        fileid = b''
    if inv.revision_id is not None:
        revid = b''.join([b' revision_id="', encode_and_escape(inv.revision_id), b'"'])
    else:
        revid = b''
    append(b'<inventory%s format="5"%s>\n' % (fileid, revid))