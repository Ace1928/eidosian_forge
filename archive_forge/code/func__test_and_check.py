import errno
import io
import logging
import multiprocessing
import os
import pickle
import resource
import socket
import stat
import subprocess
import sys
import tempfile
import time
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_concurrency import processutils
def _test_and_check(self, log_errors=None, attempts=None):
    fixture = self.useFixture(fixtures.FakeLogger(level=logging.DEBUG))
    kwargs = {}
    if log_errors:
        kwargs.update({'log_errors': log_errors})
    if attempts:
        kwargs.update({'attempts': attempts})
    err = self.assertRaises(processutils.ProcessExecutionError, processutils.execute, self.tmpfilename, **kwargs)
    self.assertEqual(41, err.exit_code)
    self.assertIn(self.tmpfilename, fixture.output)