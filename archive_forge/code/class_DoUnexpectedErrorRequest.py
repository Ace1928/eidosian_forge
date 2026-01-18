import threading
from breezy import errors, transport
from breezy.bzr.bzrdir import BzrDir
from breezy.bzr.smart import request
from breezy.tests import TestCase, TestCaseWithMemoryTransport
class DoUnexpectedErrorRequest(request.SmartServerRequest):
    """A request that encounters a generic error in self.do()"""

    def do(self):
        dict()[1]