import breezy
from .. import errors, lockdir, osutils, transport
from ..bzr.tests.test_smart import TestCaseWithSmartMedium
from ..lockable_files import LockableFiles, TransportLock
from ..transactions import (PassThroughTransaction, ReadOnlyTransaction,
from . import TestCaseInTempDir, TestNotApplicable
from .test_transactions import DummyWeave
class TestLockableFiles_RemoteLockDir(TestCaseWithSmartMedium, _TestLockableFiles_mixin):
    """LockableFile tests run with RemoteLockDir on a branch."""

    def setUp(self):
        super().setUp()
        b = self.make_branch('foo')
        self.addCleanup(b.controldir.transport.disconnect)
        self.transport = transport.get_transport_from_path('.')
        self.lockable = self.get_lockable()

    def get_lockable(self):
        branch = breezy.branch.Branch.open(self.get_url('foo'))
        self.addCleanup(branch.controldir.transport.disconnect)
        return branch.control_files