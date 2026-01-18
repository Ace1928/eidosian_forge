import breezy.branch
from breezy import branch as _mod_branch
from breezy import check, controldir, errors, gpg, osutils
from breezy import repository as _mod_repository
from breezy import revision as _mod_revision
from breezy import transport, ui, urlutils, workingtree
from breezy.bzr import bzrdir as _mod_bzrdir
from breezy.bzr.remote import (RemoteBzrDir, RemoteBzrDirFormat,
from breezy.tests import (ChrootedTestCase, TestNotApplicable, TestSkipped,
from breezy.tests.per_controldir import TestCaseWithControlDir
from breezy.transport.local import LocalTransport
from breezy.ui import CannedInputUIFactory
class TestTransportConfig(TestCaseWithControlDir):

    def test_get_config(self):
        my_dir = self.make_controldir('.')
        config = my_dir.get_config()
        try:
            config.set_default_stack_on('http://example.com')
        except errors.BzrError as e:
            if 'Cannot set config' in str(e):
                self.assertFalse(isinstance(my_dir, (_mod_bzrdir.BzrDirMeta1, RemoteBzrDir)), '%r should support configs' % my_dir)
                raise TestNotApplicable('This BzrDir format does not support configs.')
            else:
                raise
        self.assertEqual('http://example.com', config.get_default_stack_on())
        my_dir2 = controldir.ControlDir.open(self.get_url('.'))
        config2 = my_dir2.get_config()
        self.assertEqual('http://example.com', config2.get_default_stack_on())