import errno
import socket
import socketserver
import threading
from breezy import osutils, tests
from breezy.tests import test_server
from breezy.tests.scenarios import load_tests_apply_scenarios
class ShutdownConnectionHandler(test_server.TestingSmartConnectionHandler):

    def _build_protocol(self):
        self.finished = True
        return super()._build_protocol()