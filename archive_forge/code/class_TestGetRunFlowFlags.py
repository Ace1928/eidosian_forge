import json
import os.path
import shutil
import tempfile
import unittest
import mock
import six
from apitools.base.py import credentials_lib
from apitools.base.py import util
class TestGetRunFlowFlags(unittest.TestCase):

    def setUp(self):
        self._flags_actual = credentials_lib.FLAGS

    def tearDown(self):
        credentials_lib.FLAGS = self._flags_actual

    def test_with_gflags(self):
        HOST = 'myhostname'
        PORT = '144169'

        class MockFlags(object):
            auth_host_name = HOST
            auth_host_port = PORT
            auth_local_webserver = False
        credentials_lib.FLAGS = MockFlags
        flags = credentials_lib._GetRunFlowFlags(['--auth_host_name=%s' % HOST, '--auth_host_port=%s' % PORT, '--noauth_local_webserver'])
        self.assertEqual(flags.auth_host_name, HOST)
        self.assertEqual(flags.auth_host_port, PORT)
        self.assertEqual(flags.logging_level, 'ERROR')
        self.assertEqual(flags.noauth_local_webserver, True)

    def test_without_gflags(self):
        credentials_lib.FLAGS = None
        flags = credentials_lib._GetRunFlowFlags([])
        self.assertEqual(flags.auth_host_name, 'localhost')
        self.assertEqual(flags.auth_host_port, [8080, 8090])
        self.assertEqual(flags.logging_level, 'ERROR')
        self.assertEqual(flags.noauth_local_webserver, False)