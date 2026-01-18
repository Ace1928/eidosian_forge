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
def _validate_revision(self, tree, revision_id):
    """Make sure all revision entries match their checksum."""
    rev_to_sha1 = {}
    rev = self.get_revision(revision_id)
    rev_info = self.get_revision_info(revision_id)
    if not rev.revision_id == rev_info.revision_id:
        raise AssertionError()
    if not rev.revision_id == revision_id:
        raise AssertionError()
    testament = self._testament(rev, tree)
    sha1 = testament.as_sha1()
    if sha1 != rev_info.sha1:
        raise TestamentMismatch(rev.revision_id, rev_info.sha1, sha1)
    if rev.revision_id in rev_to_sha1:
        raise BzrError('Revision {%s} given twice in the list' % rev.revision_id)
    rev_to_sha1[rev.revision_id] = sha1