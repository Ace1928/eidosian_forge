import errno
import socket
import socketserver
import threading
from breezy import osutils, tests
from breezy.tests import test_server
from breezy.tests.scenarios import load_tests_apply_scenarios
class FakeServer:
    """Minimal implementation to pass to TestingSmartConnectionHandler"""
    backing_transport = None
    root_client_path = '/'