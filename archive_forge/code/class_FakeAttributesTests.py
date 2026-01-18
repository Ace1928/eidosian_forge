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
class FakeAttributesTests(SynchronousTestCase):
    """
    _Frame, _Code and _TracebackFrame objects should possess some basic
    attributes that qualify them as fake python objects, allowing the return of
    _Traceback to be used as a fake traceback. The attributes that have zero or
    empty values are there so that things expecting them find them (e.g. post
    mortem debuggers).
    """

    def test_fakeFrameAttributes(self) -> None:
        """
        L{_Frame} instances have the C{f_globals} and C{f_locals} attributes
        bound to C{dict} instance.  They also have the C{f_code} attribute
        bound to something like a code object.
        """
        back_frame = failure._Frame(('dummyparent', 'dummyparentfile', 111, None, None), None)
        fake_locals = {'local_var': 42}
        fake_globals = {'global_var': 100}
        frame = failure._Frame(('dummyname', 'dummyfilename', 42, fake_locals, fake_globals), back_frame)
        self.assertEqual(frame.f_globals, fake_globals)
        self.assertEqual(frame.f_locals, fake_locals)
        self.assertIsInstance(frame.f_code, failure._Code)
        self.assertEqual(frame.f_back, back_frame)
        self.assertIsInstance(frame.f_builtins, dict)
        self.assertIsInstance(frame.f_lasti, int)
        self.assertEqual(frame.f_lineno, 42)
        self.assertIsInstance(frame.f_trace, type(None))

    def test_fakeCodeAttributes(self) -> None:
        """
        See L{FakeAttributesTests} for more details about this test.
        """
        code = failure._Code('dummyname', 'dummyfilename')
        self.assertEqual(code.co_name, 'dummyname')
        self.assertEqual(code.co_filename, 'dummyfilename')
        self.assertIsInstance(code.co_argcount, int)
        self.assertIsInstance(code.co_code, bytes)
        self.assertIsInstance(code.co_cellvars, tuple)
        self.assertIsInstance(code.co_consts, tuple)
        self.assertIsInstance(code.co_firstlineno, int)
        self.assertIsInstance(code.co_flags, int)
        self.assertIsInstance(code.co_lnotab, bytes)
        self.assertIsInstance(code.co_freevars, tuple)
        self.assertIsInstance(code.co_posonlyargcount, int)
        self.assertIsInstance(code.co_kwonlyargcount, int)
        self.assertIsInstance(code.co_names, tuple)
        self.assertIsInstance(code.co_nlocals, int)
        self.assertIsInstance(code.co_stacksize, int)
        self.assertIsInstance(code.co_varnames, list)
        self.assertIsInstance(code.co_positions(), tuple)

    def test_fakeTracebackFrame(self) -> None:
        """
        See L{FakeAttributesTests} for more details about this test.
        """
        frame = failure._Frame(('dummyname', 'dummyfilename', 42, {}, {}), None)
        traceback_frame = failure._TracebackFrame(frame)
        self.assertEqual(traceback_frame.tb_frame, frame)
        self.assertEqual(traceback_frame.tb_lineno, 42)
        self.assertIsInstance(traceback_frame.tb_lasti, int)
        self.assertTrue(hasattr(traceback_frame, 'tb_next'))