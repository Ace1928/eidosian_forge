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
def _load_one_rev(self, rev_id):
    """Load a revision object into memory.

        Any parents not either loaded or abandoned get queued to be
        loaded."""
    self.pb.update(gettext('loading revision'), len(self.revisions), len(self.known_revisions))
    if not self.branch.repository.has_revision(rev_id):
        self.pb.clear()
        ui.ui_factory.note(gettext('revision {%s} not present in branch; will be converted as a ghost') % rev_id)
        self.absent_revisions.add(rev_id)
    else:
        rev = self.branch.repository.get_revision(rev_id)
        for parent_id in rev.parent_ids:
            self.known_revisions.add(parent_id)
            self.to_read.append(parent_id)
        self.revisions[rev_id] = rev