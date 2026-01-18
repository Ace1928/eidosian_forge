import base64
import os
import pprint
from io import BytesIO
from ... import cache_utf8, osutils, timestamp
from ...errors import BzrError, NoSuchId, TestamentMismatch
from ...osutils import pathjoin, sha_string, sha_strings
from ...revision import NULL_REVISION, Revision
from ...trace import mutter, warning
from ...tree import InterTree, Tree
from ..inventory import (Inventory, InventoryDirectory, InventoryFile,
from ..inventorytree import InventoryTree
from ..testament import StrictTestament
from ..xml5 import serializer_v5
from . import apply_bundle
def _validate_inventory(self, inv, revision_id):
    """At this point we should have generated the BundleTree,
        so build up an inventory, and make sure the hashes match.
        """
    cs = serializer_v5.write_inventory_to_chunks(inv)
    sha1 = sha_strings(cs)
    rev = self.get_revision(revision_id)
    if rev.revision_id != revision_id:
        raise AssertionError()
    if sha1 != rev.inventory_sha1:
        with open(',,bogus-inv', 'wb') as f:
            f.writelines(cs)
        warning('Inventory sha hash mismatch for revision %s. %s != %s' % (revision_id, sha1, rev.inventory_sha1))