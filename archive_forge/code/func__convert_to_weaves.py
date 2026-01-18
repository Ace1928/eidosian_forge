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
def _convert_to_weaves(self):
    ui.ui_factory.note(gettext('note: upgrade may be faster if all store files are ungzipped first'))
    try:
        stat = self.controldir.transport.stat('weaves')
        if not S_ISDIR(stat.st_mode):
            self.controldir.transport.delete('weaves')
            self.controldir.transport.mkdir('weaves')
    except NoSuchFile:
        self.controldir.transport.mkdir('weaves')
    self.inv_weave = weave.Weave('inventory')
    self.text_weaves = {}
    self.controldir.transport.delete('branch-format')
    self.branch = self.controldir.open_branch()
    self._convert_working_inv()
    rev_history = self.branch._revision_history()
    self.known_revisions = set(rev_history)
    self.to_read = rev_history[-1:]
    while self.to_read:
        rev_id = self.to_read.pop()
        if rev_id not in self.revisions and rev_id not in self.absent_revisions:
            self._load_one_rev(rev_id)
    self.pb.clear()
    to_import = self._make_order()
    for i, rev_id in enumerate(to_import):
        self.pb.update(gettext('converting revision'), i, len(to_import))
        self._convert_one_rev(rev_id)
    self.pb.clear()
    self._write_all_weaves()
    self._write_all_revs()
    ui.ui_factory.note(gettext('upgraded to weaves:'))
    ui.ui_factory.note('  ' + gettext('%6d revisions and inventories') % len(self.revisions))
    ui.ui_factory.note('  ' + gettext('%6d revisions not present') % len(self.absent_revisions))
    ui.ui_factory.note('  ' + gettext('%6d texts') % self.text_count)
    self._cleanup_spare_files_after_format4()
    self.branch._transport.put_bytes('branch-format', BzrDirFormat5().get_format_string(), mode=self.controldir._get_file_mode())