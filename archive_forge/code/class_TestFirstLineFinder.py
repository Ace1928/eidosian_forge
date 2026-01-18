import unittest
import inspect
from numba import njit
from numba.tests.support import TestCase
from numba.misc.firstlinefinder import get_func_body_first_lineno
class TestFirstLineFinder(TestCase):
    """
    The following methods contains tests that are sensitive to the source
    locations w.r.t. the beginning of each method.
    """

    def _get_grandparent_caller_code(self):
        frame = inspect.currentframe()
        caller_frame = inspect.getouterframes(frame)
        return caller_frame[2].frame.f_code

    def assert_line_location(self, expected, offset_from_caller):
        grandparent_co = self._get_grandparent_caller_code()
        lno = grandparent_co.co_firstlineno
        self.assertEqual(expected, lno + offset_from_caller)

    def test_decorated_odd_comment_indent(self):

        @njit
        def foo():
            return 1
        first_def_line = get_func_body_first_lineno(foo)
        self.assert_line_location(first_def_line, 4)

    def test_undecorated_odd_comment_indent(self):

        def foo():
            return 1
        first_def_line = get_func_body_first_lineno(njit(foo))
        self.assert_line_location(first_def_line, 3)

    def test_unnamed_lambda(self):
        foo = lambda: 1
        first_def_line = get_func_body_first_lineno(njit(foo))
        self.assertIsNone(first_def_line)

    def test_nested_function(self):

        def foo():

            @njit
            def foo():
                return 1
            return foo
        inner = foo()
        first_def_line = get_func_body_first_lineno(inner)
        self.assert_line_location(first_def_line, 5)

    def test_pass_statement(self):

        @njit
        def foo():
            pass
        first_def_line = get_func_body_first_lineno(foo)
        self.assert_line_location(first_def_line, 3)

    def test_string_eval(self):
        source = 'def foo():\n            pass\n        '
        globalns = {}
        exec(source, globalns)
        foo = globalns['foo']
        first_def_line = get_func_body_first_lineno(foo)
        self.assertIsNone(first_def_line)

    def test_single_line_function(self):

        @njit
        def foo():
            pass
        first_def_line = get_func_body_first_lineno(foo)
        self.assert_line_location(first_def_line, 2)

    def test_docstring(self):

        @njit
        def foo():
            """Docstring
            """
            pass
        first_def_line = get_func_body_first_lineno(foo)
        self.assert_line_location(first_def_line, 5)

    def test_docstring_2(self):

        @njit
        def foo():
            """Docstring
            """
            'Not Docstring, but a bare string literal\n            '
            pass
        first_def_line = get_func_body_first_lineno(foo)
        self.assert_line_location(first_def_line, 5)