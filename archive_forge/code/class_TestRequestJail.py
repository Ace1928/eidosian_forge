import threading
from breezy import errors, transport
from breezy.bzr.bzrdir import BzrDir
from breezy.bzr.smart import request
from breezy.tests import TestCase, TestCaseWithMemoryTransport
class TestRequestJail(TestCaseWithMemoryTransport):

    def test_jail(self):
        transport = self.get_transport('blah')
        req = request.SmartServerRequest(transport)
        self.assertEqual(None, request.jail_info.transports)
        req.setup_jail()
        self.assertEqual([transport], request.jail_info.transports)
        req.teardown_jail()
        self.assertEqual(None, request.jail_info.transports)