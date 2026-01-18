import threading
from breezy import errors, transport
from breezy.bzr.bzrdir import BzrDir
from breezy.bzr.smart import request
from breezy.tests import TestCase, TestCaseWithMemoryTransport
class CheckJailRequest(request.SmartServerRequest):

    def __init__(self, *args):
        request.SmartServerRequest.__init__(self, *args)
        self.jail_transports_log = []

    def do(self):
        self.jail_transports_log.append(request.jail_info.transports)

    def do_chunk(self, bytes):
        self.jail_transports_log.append(request.jail_info.transports)

    def do_end(self):
        self.jail_transports_log.append(request.jail_info.transports)