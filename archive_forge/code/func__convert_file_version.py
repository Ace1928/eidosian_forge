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
def _convert_file_version(self, rev, ie, parent_invs):
    """Convert one version of one file.

        The file needs to be added into the weave if it is a merge
        of >=2 parents or if it's changed from its parent.
        """
    file_id = ie.file_id
    rev_id = rev.revision_id
    w = self.text_weaves.get(file_id)
    if w is None:
        w = weave.Weave(file_id)
        self.text_weaves[file_id] = w
    text_changed = False
    parent_candiate_entries = ie.parent_candidates(parent_invs)
    heads = graph.Graph(self).heads(parent_candiate_entries)
    previous_entries = {head: parent_candiate_entries[head] for head in heads}
    self.snapshot_ie(previous_entries, ie, w, rev_id)