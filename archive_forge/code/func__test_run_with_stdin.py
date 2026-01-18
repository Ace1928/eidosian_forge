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
def _test_run_with_stdin(self, expect_byte=True):
    code, out, err = self.execute(['cat'], stdin=b'teststr')
    self.assertEqual(0, code)
    if expect_byte:
        expect_out = b'teststr'
        expect_err = b''
    else:
        expect_out = 'teststr'
        expect_err = ''
    self.assertEqual(expect_out, out)
    self.assertEqual(expect_err, err)