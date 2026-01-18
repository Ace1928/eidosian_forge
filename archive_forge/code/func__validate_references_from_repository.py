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
def _validate_references_from_repository(self, repository):
    """Now that we have a repository which should have some of the
        revisions we care about, go through and validate all of them
        that we can.
        """
    rev_to_sha = {}
    inv_to_sha = {}

    def add_sha(d, revision_id, sha1):
        if revision_id is None:
            if sha1 is not None:
                raise BzrError('A Null revision should alwayshave a null sha1 hash')
            return
        if revision_id in d:
            if sha1 != d[revision_id]:
                raise BzrError('** Revision %r referenced with 2 different sha hashes %s != %s' % (revision_id, sha1, d[revision_id]))
        else:
            d[revision_id] = sha1
    checked = {}
    for rev_info in self.revisions:
        checked[rev_info.revision_id] = True
        add_sha(rev_to_sha, rev_info.revision_id, rev_info.sha1)
    for rev, rev_info in zip(self.real_revisions, self.revisions):
        add_sha(inv_to_sha, rev_info.revision_id, rev_info.inventory_sha1)
    count = 0
    missing = {}
    for revision_id, sha1 in rev_to_sha.items():
        if repository.has_revision(revision_id):
            testament = StrictTestament.from_revision(repository, revision_id)
            local_sha1 = self._testament_sha1_from_revision(repository, revision_id)
            if sha1 != local_sha1:
                raise BzrError('sha1 mismatch. For revision id {%s}local: %s, bundle: %s' % (revision_id, local_sha1, sha1))
            else:
                count += 1
        elif revision_id not in checked:
            missing[revision_id] = sha1
    if len(missing) > 0:
        warning('Not all revision hashes could be validated. Unable validate %d hashes' % len(missing))
    mutter('Verified %d sha hashes for the bundle.' % count)
    self._validated_revisions_against_repo = True