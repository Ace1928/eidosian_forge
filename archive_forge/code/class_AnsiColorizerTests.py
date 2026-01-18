import errno
import os
import re
import sys
from inspect import getmro
from io import BytesIO, StringIO
from typing import Type
from unittest import (
from hamcrest import assert_that, equal_to, has_item, has_length
from twisted.python import log
from twisted.python.failure import Failure
from twisted.trial import itrial, reporter, runner, unittest, util
from twisted.trial.reporter import UncleanWarningsReporterWrapper, _ExitWrapper
from twisted.trial.test import erroneous, sample
from twisted.trial.unittest import SkipTest, Todo, makeTodo
from .._dist.test.matchers import isFailure, matches_result, similarFrame
from .matchers import after
class AnsiColorizerTests(unittest.SynchronousTestCase):
    """
    Tests for L{reporter._AnsiColorizer}.
    """

    def setUp(self):
        self.savedModules = sys.modules.copy()

    def tearDown(self):
        sys.modules.clear()
        sys.modules.update(self.savedModules)

    def test_supportedStdOutTTY(self):
        """
        L{reporter._AnsiColorizer.supported} returns C{False} if the given
        stream is not a TTY.
        """
        self.assertFalse(reporter._AnsiColorizer.supported(FakeStream(False)))

    def test_supportedNoCurses(self):
        """
        L{reporter._AnsiColorizer.supported} returns C{False} if the curses
        module can't be imported.
        """
        sys.modules['curses'] = None
        self.assertFalse(reporter._AnsiColorizer.supported(FakeStream()))

    def test_supportedSetupTerm(self):
        """
        L{reporter._AnsiColorizer.supported} returns C{True} if
        C{curses.tigetnum} returns more than 2 supported colors. It only tries
        to call C{curses.setupterm} if C{curses.tigetnum} previously failed
        with a C{curses.error}.
        """

        class fakecurses:
            error = RuntimeError
            setUp = 0

            def setupterm(self):
                self.setUp += 1

            def tigetnum(self, value):
                if self.setUp:
                    return 3
                else:
                    raise self.error()
        sys.modules['curses'] = fakecurses()
        self.assertTrue(reporter._AnsiColorizer.supported(FakeStream()))
        self.assertTrue(reporter._AnsiColorizer.supported(FakeStream()))
        self.assertEqual(sys.modules['curses'].setUp, 1)

    def test_supportedTigetNumWrongError(self):
        """
        L{reporter._AnsiColorizer.supported} returns C{False} and doesn't try
        to call C{curses.setupterm} if C{curses.tigetnum} returns something
        different than C{curses.error}.
        """

        class fakecurses:
            error = RuntimeError

            def tigetnum(self, value):
                raise ValueError()
        sys.modules['curses'] = fakecurses()
        self.assertFalse(reporter._AnsiColorizer.supported(FakeStream()))

    def test_supportedTigetNumNotEnoughColor(self):
        """
        L{reporter._AnsiColorizer.supported} returns C{False} if
        C{curses.tigetnum} returns less than 2 supported colors.
        """

        class fakecurses:
            error = RuntimeError

            def tigetnum(self, value):
                return 1
        sys.modules['curses'] = fakecurses()
        self.assertFalse(reporter._AnsiColorizer.supported(FakeStream()))

    def test_supportedTigetNumErrors(self):
        """
        L{reporter._AnsiColorizer.supported} returns C{False} if
        C{curses.tigetnum} raises an error, and calls C{curses.setupterm} once.
        """

        class fakecurses:
            error = RuntimeError
            setUp = 0

            def setupterm(self):
                self.setUp += 1

            def tigetnum(self, value):
                raise self.error()
        sys.modules['curses'] = fakecurses()
        self.assertFalse(reporter._AnsiColorizer.supported(FakeStream()))
        self.assertEqual(sys.modules['curses'].setUp, 1)