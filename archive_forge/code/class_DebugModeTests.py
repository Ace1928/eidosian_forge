from __future__ import annotations
import linecache
import pdb
import re
import sys
import traceback
from dis import distb
from io import StringIO
from traceback import FrameSummary
from types import TracebackType
from typing import Any, Generator
from unittest import skipIf
from cython_test_exception_raiser import raiser
from twisted.python import failure, reflect
from twisted.trial.unittest import SynchronousTestCase
class DebugModeTests(SynchronousTestCase):
    """
    Failure's debug mode should allow jumping into the debugger.
    """

    def setUp(self) -> None:
        """
        Override pdb.post_mortem so we can make sure it's called.
        """
        post_mortem = pdb.post_mortem
        origInit = failure.Failure.__init__

        def restore() -> None:
            pdb.post_mortem = post_mortem
            failure.Failure.__init__ = origInit
        self.addCleanup(restore)
        self.result: list[TracebackType | None] = []

        def logging_post_mortem(t: TracebackType | None=None) -> None:
            self.result.append(t)
        pdb.post_mortem = logging_post_mortem
        failure.startDebugMode()

    def test_regularFailure(self) -> None:
        """
        If startDebugMode() is called, calling Failure() will first call
        pdb.post_mortem with the traceback.
        """
        try:
            1 / 0
        except BaseException:
            typ, exc, tb = sys.exc_info()
            f = failure.Failure()
        self.assertEqual(self.result, [tb])
        self.assertFalse(f.captureVars)

    def test_captureVars(self) -> None:
        """
        If startDebugMode() is called, passing captureVars to Failure() will
        not blow up.
        """
        try:
            1 / 0
        except BaseException:
            typ, exc, tb = sys.exc_info()
            f = failure.Failure(captureVars=True)
        self.assertEqual(self.result, [tb])
        self.assertTrue(f.captureVars)