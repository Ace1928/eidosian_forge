from __future__ import absolute_import
import threading
import warnings
import subprocess
import sys
from unittest import SkipTest, TestCase
import twisted
from twisted.python.log import PythonLoggingObserver
from twisted.python import log
from twisted.python.runtime import platform
from twisted.internet.task import Clock
from .._eventloop import EventLoop, ThreadLogObserver, _store
from ..tests import crochet_directory
import sys
import crochet
import sys
from logging import StreamHandler, Formatter, getLogger, DEBUG
import crochet
from twisted.python import log
from twisted.logger import Logger
import time
class ReactorImportTests(TestCase):
    """
    Tests for when the reactor gets imported.

    The reactor should only be imported as part of setup()/no_setup(),
    rather than as side-effect of Crochet import, since daemonization
    doesn't work if reactor is imported
    (https://twistedmatrix.com/trac/ticket/7105).
    """

    def test_crochet_import_no_reactor(self):
        """
        Importing crochet should not import the reactor.
        """
        program = 'import sys\nimport crochet\n\nif "twisted.internet.reactor" not in sys.modules:\n    sys.exit(23)\n'
        process = subprocess.Popen([sys.executable, '-c', program], cwd=crochet_directory)
        self.assertEqual(process.wait(), 23)