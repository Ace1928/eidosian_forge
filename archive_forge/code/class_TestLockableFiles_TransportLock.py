import breezy
from .. import errors, lockdir, osutils, transport
from ..bzr.tests.test_smart import TestCaseWithSmartMedium
from ..lockable_files import LockableFiles, TransportLock
from ..transactions import (PassThroughTransaction, ReadOnlyTransaction,
from . import TestCaseInTempDir, TestNotApplicable
from .test_transactions import DummyWeave
class TestLockableFiles_TransportLock(TestCaseInTempDir, _TestLockableFiles_mixin):

    def setUp(self):
        super().setUp()
        t = transport.get_transport_from_path('.')
        t.mkdir('.bzr')
        self.sub_transport = t.clone('.bzr')
        self.lockable = self.get_lockable()
        self.lockable.create_lock()

    def stop_server(self):
        super().stop_server()
        try:
            del self.sub_transport
        except AttributeError:
            pass

    def get_lockable(self):
        return LockableFiles(self.sub_transport, 'my-lock', TransportLock)