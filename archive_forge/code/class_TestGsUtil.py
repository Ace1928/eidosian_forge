from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import importlib
import unittest
from unittest import mock
from google.auth import exceptions as google_auth_exceptions
from gslib.command_runner import CommandRunner
from gslib.utils import system_util
import gslib
import gslib.tests.testcase as testcase
class TestGsUtil(testcase.GsUtilIntegrationTestCase):
    """Integration tests for top-level gsutil command."""

    def test_long_version_arg(self):
        stdout = self.RunGsUtil(['--version'], return_stdout=True)
        self.assertEqual('gsutil version: %s\n' % gslib.VERSION, stdout)

    def test_version_command(self):
        stdout = self.RunGsUtil(['version'], return_stdout=True)
        self.assertEqual('gsutil version: %s\n' % gslib.VERSION, stdout)

    def test_version_long(self):
        stdout = self.RunGsUtil(['version', '-l'], return_stdout=True)
        self.assertIn('gsutil version: %s\n' % gslib.VERSION, stdout)
        self.assertIn('boto version', stdout)
        self.assertIn('checksum', stdout)
        self.assertIn('config path', stdout)
        self.assertIn('gsutil path', stdout)