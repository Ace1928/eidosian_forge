import contextlib
import io
import logging
import os
import pwd
import shutil
import signal
import sys
import threading
import time
from unittest import mock
import fixtures
import testtools
from testtools import content
from oslo_rootwrap import client
from oslo_rootwrap import cmd
from oslo_rootwrap import subprocess
from oslo_rootwrap.tests import run_daemon
class RootwrapTest(_FunctionalBase, testtools.TestCase):

    def setUp(self):
        super(RootwrapTest, self).setUp()
        self.cmd = [sys.executable, '-c', 'from oslo_rootwrap import cmd; cmd.main()', self.config_file]

    def execute(self, cmd, stdin=None):
        proc = subprocess.Popen(self.cmd + cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate(stdin)
        self.addDetail('stdout', content.text_content(out.decode('utf-8', 'replace')))
        self.addDetail('stderr', content.text_content(err.decode('utf-8', 'replace')))
        return (proc.returncode, out, err)

    def test_run_once(self):
        self._test_run_once(expect_byte=True)

    def test_run_with_stdin(self):
        self._test_run_with_stdin(expect_byte=True)