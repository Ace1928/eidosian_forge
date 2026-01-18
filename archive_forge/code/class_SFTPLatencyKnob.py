import os
import socket
import sys
import time
from breezy import config, controldir, errors, tests
from breezy import transport as _mod_transport
from breezy import ui
from breezy.osutils import lexists
from breezy.tests import TestCase, TestCaseWithTransport, TestSkipped, features
from breezy.tests.http_server import HttpServer
class SFTPLatencyKnob(TestCaseWithSFTPServer):
    """Test that the testing SFTPServer's latency knob works."""

    def test_latency_knob_slows_transport(self):
        start_time = time.time()
        self.get_server().add_latency = 0.5
        transport = self.get_transport()
        transport.has('not me')
        with_latency_knob_time = time.time() - start_time
        self.assertTrue(with_latency_knob_time > 0.4)

    def test_default(self):
        raise TestSkipped('Timing-sensitive test')
        start_time = time.time()
        transport = self.get_transport()
        transport.has('not me')
        regular_time = time.time() - start_time
        self.assertTrue(regular_time < 0.5)