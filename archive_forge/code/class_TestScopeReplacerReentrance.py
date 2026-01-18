import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
class TestScopeReplacerReentrance(TestCase):
    """The ScopeReplacer should be reentrant.

    Invoking a replacer while an invocation was already on-going leads to a
    race to see which invocation will be the first to call _replace.
    The losing caller used to see an exception (bugs 396819 and 702914).

    These tests set up a tracer that stops at a suitable moment (upon
    entry of a specified method) and starts another call to the
    functionality in question (__call__, __getattribute__, __setattr_)
    in order to win the race, setting up the original caller to lose.
    """

    def tracer(self, frame, event, arg):
        if event != 'call':
            return self.tracer
        code = frame.f_code
        filename = code.co_filename
        filename = re.sub('\\.py[co]$', '.py', filename)
        function_name = code.co_name
        if filename.endswith('lazy_import.py') and function_name == self.method_to_trace:
            sys.settrace(None)
            self.racer()
        return self.tracer

    def run_race(self, racer, method_to_trace='_resolve'):
        self.overrideAttr(lazy_import.ScopeReplacer, '_should_proxy', True)
        self.racer = racer
        self.method_to_trace = method_to_trace
        sys.settrace(self.tracer)
        self.racer()
        self.assertEqual(None, sys.gettrace())

    def test_call(self):

        def factory(*args):
            return factory
        replacer = lazy_import.ScopeReplacer({}, factory, 'name')
        self.run_race(replacer)

    def test_setattr(self):

        class Replaced:
            pass

        def factory(*args):
            return Replaced()
        replacer = lazy_import.ScopeReplacer({}, factory, 'name')

        def racer():
            replacer.foo = 42
        self.run_race(racer)

    def test_getattribute(self):

        class Replaced:
            foo = 'bar'

        def factory(*args):
            return Replaced()
        replacer = lazy_import.ScopeReplacer({}, factory, 'name')

        def racer():
            replacer.foo
        self.run_race(racer)