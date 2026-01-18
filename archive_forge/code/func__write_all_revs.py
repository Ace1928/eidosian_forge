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
def _write_all_revs(self):
    """Write all revisions out in new form."""
    self.controldir.transport.delete_tree('revision-store')
    self.controldir.transport.mkdir('revision-store')
    revision_transport = self.controldir.transport.clone('revision-store')
    from ...bzr.xml5 import serializer_v5
    from .repository import RevisionTextStore
    revision_store = RevisionTextStore(revision_transport, serializer_v5, False, versionedfile.PrefixMapper(), lambda: True, lambda: True)
    try:
        for i, rev_id in enumerate(self.converted_revs):
            self.pb.update(gettext('write revision'), i, len(self.converted_revs))
            lines = serializer_v5.write_revision_to_lines(self.revisions[rev_id])
            key = (rev_id,)
            revision_store.add_lines(key, None, lines)
    finally:
        self.pb.clear()