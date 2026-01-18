from typing import List
from ...bzr import inventory
from ...bzr.inventory import ROOT_ID, Inventory
from ...bzr.xml_serializer import (Element, SubElement, XMLSerializer,
from ...errors import BzrError
from ...revision import Revision
def _unpack_entry(self, elt, entry_cache=None, return_from_cache=False):
    parent_id = elt.get('parent_id')
    parent_id = parent_id.encode('ascii') if parent_id else ROOT_ID
    file_id = elt.get('file_id').encode('ascii')
    kind = elt.get('kind')
    if kind == 'directory':
        ie = inventory.InventoryDirectory(file_id, elt.get('name'), parent_id)
    elif kind == 'file':
        ie = inventory.InventoryFile(file_id, elt.get('name'), parent_id)
        ie.text_id = elt.get('text_id')
        if ie.text_id is not None:
            ie.text_id = ie.text_id.encode('utf-8')
        ie.text_sha1 = elt.get('text_sha1')
        if ie.text_sha1 is not None:
            ie.text_sha1 = ie.text_sha1.encode('ascii')
        v = elt.get('text_size')
        ie.text_size = v and int(v)
    elif kind == 'symlink':
        ie = inventory.InventoryLink(file_id, elt.get('name'), parent_id)
        ie.symlink_target = elt.get('symlink_target')
    else:
        raise BzrError('unknown kind %r' % kind)
    return ie