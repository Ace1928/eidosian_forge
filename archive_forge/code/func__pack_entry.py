from typing import List
from ...bzr import inventory
from ...bzr.inventory import ROOT_ID, Inventory
from ...bzr.xml_serializer import (Element, SubElement, XMLSerializer,
from ...errors import BzrError
from ...revision import Revision
def _pack_entry(self, ie):
    """Convert InventoryEntry to XML element"""
    e = Element('entry')
    e.set('name', ie.name)
    e.set('file_id', ie.file_id.decode('ascii'))
    e.set('kind', ie.kind)
    if ie.text_size is not None:
        e.set('text_size', '%d' % ie.text_size)
    for f in ['text_id', 'text_sha1', 'symlink_target']:
        v = getattr(ie, f)
        if v is not None:
            e.set(f, v)
    if ie.parent_id != ROOT_ID:
        e.set('parent_id', ie.parent_id)
    e.tail = '\n'
    return e