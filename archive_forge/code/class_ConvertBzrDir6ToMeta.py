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
class ConvertBzrDir6ToMeta(Converter):
    """Converts format 6 bzr dirs to metadirs."""

    def convert(self, to_convert, pb):
        """See Converter.convert()."""
        from ...bzr.fullhistory import BzrBranchFormat5
        from .repository import RepositoryFormat7
        self.controldir = to_convert
        self.pb = ui.ui_factory.nested_progress_bar()
        self.count = 0
        self.total = 20
        self.garbage_inventories = []
        self.dir_mode = self.controldir._get_dir_mode()
        self.file_mode = self.controldir._get_file_mode()
        ui.ui_factory.note(gettext('starting upgrade from format 6 to metadir'))
        self.controldir.transport.put_bytes('branch-format', b'Converting to format 6', mode=self.file_mode)
        try:
            self.step(gettext('Removing ancestry.weave'))
            self.controldir.transport.delete('ancestry.weave')
        except NoSuchFile:
            pass
        self.step(gettext('Finding branch files'))
        last_revision = self.controldir.open_branch().last_revision()
        bzrcontents = self.controldir.transport.list_dir('.')
        for name in bzrcontents:
            if name.startswith('basis-inventory.'):
                self.garbage_inventories.append(name)
        repository_names = [('inventory.weave', True), ('revision-store', True), ('weaves', True)]
        self.step(gettext('Upgrading repository') + '  ')
        self.controldir.transport.mkdir('repository', mode=self.dir_mode)
        self.make_lock('repository')
        self.put_format('repository', RepositoryFormat7())
        for entry in repository_names:
            self.move_entry('repository', entry)
        self.step(gettext('Upgrading branch') + '      ')
        self.controldir.transport.mkdir('branch', mode=self.dir_mode)
        self.make_lock('branch')
        self.put_format('branch', BzrBranchFormat5())
        branch_files = [('revision-history', True), ('branch-name', True), ('parent', False)]
        for entry in branch_files:
            self.move_entry('branch', entry)
        checkout_files = [('pending-merges', True), ('inventory', True), ('stat-cache', False)]
        for name, mandatory in checkout_files:
            if mandatory and name not in bzrcontents:
                has_checkout = False
                break
        else:
            has_checkout = True
        if not has_checkout:
            ui.ui_factory.note(gettext('No working tree.'))
            for name, mandatory in checkout_files:
                if name in bzrcontents:
                    self.controldir.transport.delete(name)
        else:
            from ...bzr.workingtree_3 import WorkingTreeFormat3
            self.step(gettext('Upgrading working tree'))
            self.controldir.transport.mkdir('checkout', mode=self.dir_mode)
            self.make_lock('checkout')
            self.put_format('checkout', WorkingTreeFormat3())
            for path in self.garbage_inventories:
                self.controldir.transport.delete(path)
            for entry in checkout_files:
                self.move_entry('checkout', entry)
            if last_revision is not None:
                self.controldir.transport.put_bytes('checkout/last-revision', last_revision)
        self.controldir.transport.put_bytes('branch-format', BzrDirMetaFormat1().get_format_string(), mode=self.file_mode)
        self.pb.finished()
        return ControlDir.open(self.controldir.user_url)

    def make_lock(self, name):
        """Make a lock for the new control dir name."""
        self.step(gettext('Make %s lock') % name)
        ld = lockdir.LockDir(self.controldir.transport, '%s/lock' % name, file_modebits=self.file_mode, dir_modebits=self.dir_mode)
        ld.create()

    def move_entry(self, new_dir, entry):
        """Move then entry name into new_dir."""
        name = entry[0]
        mandatory = entry[1]
        self.step(gettext('Moving %s') % name)
        try:
            self.controldir.transport.move(name, '{}/{}'.format(new_dir, name))
        except NoSuchFile:
            if mandatory:
                raise

    def put_format(self, dirname, format):
        self.controldir.transport.put_bytes('%s/format' % dirname, format.get_format_string(), self.file_mode)