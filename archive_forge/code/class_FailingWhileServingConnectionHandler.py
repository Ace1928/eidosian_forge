import errno
import socket
import socketserver
import threading
from breezy import osutils, tests
from breezy.tests import test_server
from breezy.tests.scenarios import load_tests_apply_scenarios
class FailingWhileServingConnectionHandler(TCPConnectionHandler):

    def handle(request):
        self.connection_thread = threading.currentThread()
        self.connection_thread.set_sync_event(caught)
        raise CantServe()