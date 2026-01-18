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
def _convert_revision_contents(self, rev, inv, present_parents):
    """Convert all the files within a revision.

        Also upgrade the inventory to refer to the text revision ids."""
    rev_id = rev.revision_id
    trace.mutter('converting texts of revision {%s}', rev_id)
    parent_invs = list(map(self._load_updated_inventory, present_parents))
    entries = inv.iter_entries()
    next(entries)
    for path, ie in entries:
        self._convert_file_version(rev, ie, parent_invs)