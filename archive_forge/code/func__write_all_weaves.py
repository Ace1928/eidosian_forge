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
def _write_all_weaves(self):
    controlweaves = VersionedFileStore(self.controldir.transport, prefixed=False, versionedfile_class=weave.WeaveFile)
    weave_transport = self.controldir.transport.clone('weaves')
    weaves = VersionedFileStore(weave_transport, prefixed=False, versionedfile_class=weave.WeaveFile)
    transaction = WriteTransaction()
    try:
        i = 0
        for file_id, file_weave in self.text_weaves.items():
            self.pb.update(gettext('writing weave'), i, len(self.text_weaves))
            weaves._put_weave(file_id, file_weave, transaction)
            i += 1
        self.pb.update(gettext('inventory'), 0, 1)
        controlweaves._put_weave(b'inventory', self.inv_weave, transaction)
        self.pb.update(gettext('inventory'), 1, 1)
    finally:
        self.pb.clear()