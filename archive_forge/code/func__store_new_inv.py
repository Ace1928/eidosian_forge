from io import BytesIO
from ... import errors, lockable_files
from ...bzr.bzrdir import BzrDir, BzrDirFormat, BzrDirMetaFormat1
from ...controldir import (ControlDir, Converter, MustHaveWorkingTree,
from ...i18n import gettext
from ...lazy_import import lazy_import
from ...transport import NoSuchFile, get_transport, local
import os
from breezy import (
from breezy.bzr import (
from breezy.plugins.weave_fmt.store.versioned import VersionedFileStore
from breezy.transactions import WriteTransaction
from breezy.plugins.weave_fmt import xml4
def _store_new_inv(self, rev, inv, present_parents):
    new_inv_xml = xml5.serializer_v5.write_inventory_to_lines(inv)
    new_inv_sha1 = osutils.sha_strings(new_inv_xml)
    self.inv_weave.add_lines(rev.revision_id, present_parents, new_inv_xml)
    rev.inventory_sha1 = new_inv_sha1