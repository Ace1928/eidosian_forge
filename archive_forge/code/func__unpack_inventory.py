from typing import List
from ...bzr import inventory
from ...bzr.inventory import ROOT_ID, Inventory
from ...bzr.xml_serializer import (Element, SubElement, XMLSerializer,
from ...errors import BzrError
from ...revision import Revision
def _unpack_inventory(self, elt, revision_id=None, entry_cache=None, return_from_cache=False):
    """Construct from XML Element

        :param revision_id: Ignored parameter used by xml5.
        """
    root_id = elt.get('file_id')
    root_id = root_id.encode('ascii') if root_id else ROOT_ID
    inv = Inventory(root_id)
    for e in elt:
        ie = self._unpack_entry(e, entry_cache=entry_cache, return_from_cache=return_from_cache)
        if ie.parent_id == ROOT_ID:
            ie.parent_id = root_id
        inv.add(ie)
    return inv