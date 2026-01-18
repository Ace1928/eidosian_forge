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
class ProcessSetupTests(TestCase):
    """
    setup() enables support for IReactorProcess on POSIX plaforms.
    """

    def test_posix(self):
        """
        On POSIX systems, setup() installs a LoopingCall that runs
        t.i.process.reapAllProcesses() 10 times a second.
        """
        if platform.type != 'posix':
            raise SkipTest('SIGCHLD is a POSIX-specific issue')
        reactor = FakeReactor()
        reaps = []
        s = EventLoop(lambda: reactor, lambda f, *g: None, reapAllProcesses=lambda: reaps.append(1))
        s.setup()
        reactor.advance(0.1)
        self.assertEqual(reaps, [1])
        reactor.advance(0.1)
        self.assertEqual(reaps, [1, 1])
        reactor.advance(0.1)
        self.assertEqual(reaps, [1, 1, 1])

    def test_non_posix(self):
        """
        On non-POSIX systems, setup() does not install a LoopingCall.
        """
        if platform.type == 'posix':
            raise SkipTest('This test is for non-POSIX systems.')
        reactor = FakeReactor()
        s = EventLoop(lambda: reactor, lambda f, *g: None)
        s.setup()
        self.assertFalse(reactor.getDelayedCalls())