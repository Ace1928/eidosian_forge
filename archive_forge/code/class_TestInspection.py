import inspect
import os
import sys
import unittest
from collections.abc import Sequence
from typing import List
from bpython import inspection
from bpython.test.fodder import encoding_ascii
from bpython.test.fodder import encoding_latin1
from bpython.test.fodder import encoding_utf8
class TestInspection(unittest.TestCase):

    def test_parsekeywordpairs(self):

        def fails(spam=['-a', '-b']):
            pass
        argspec = inspection.getfuncprops('fails', fails)
        self.assertIsNotNone(argspec)
        defaults = argspec.argspec.defaults
        self.assertEqual(str(defaults[0]), '["-a", "-b"]')

    def test_pasekeywordpairs_string(self):

        def spam(eggs='foo, bar'):
            pass
        defaults = inspection.getfuncprops('spam', spam).argspec.defaults
        self.assertEqual(repr(defaults[0]), '"foo, bar"')

    def test_parsekeywordpairs_multiple_keywords(self):

        def spam(eggs=23, foobar='yay'):
            pass
        defaults = inspection.getfuncprops('spam', spam).argspec.defaults
        self.assertEqual(repr(defaults[0]), '23')
        self.assertEqual(repr(defaults[1]), '"yay"')

    def test_pasekeywordpairs_annotation(self):

        def spam(eggs: str='foo, bar'):
            pass
        defaults = inspection.getfuncprops('spam', spam).argspec.defaults
        self.assertEqual(repr(defaults[0]), '"foo, bar"')

    def test_get_encoding_ascii(self):
        self.assertEqual(inspection.get_encoding(encoding_ascii), 'ascii')
        self.assertEqual(inspection.get_encoding(encoding_ascii.foo), 'ascii')

    def test_get_encoding_latin1(self):
        self.assertEqual(inspection.get_encoding(encoding_latin1), 'latin1')
        self.assertEqual(inspection.get_encoding(encoding_latin1.foo), 'latin1')

    def test_get_encoding_utf8(self):
        self.assertEqual(inspection.get_encoding(encoding_utf8), 'utf-8')
        self.assertEqual(inspection.get_encoding(encoding_utf8.foo), 'utf-8')

    def test_get_source_ascii(self):
        self.assertEqual(inspect.getsource(encoding_ascii.foo), foo_ascii_only)

    def test_get_source_utf8(self):
        self.assertEqual(inspect.getsource(encoding_utf8.foo), foo_non_ascii)

    def test_get_source_latin1(self):
        self.assertEqual(inspect.getsource(encoding_latin1.foo), foo_non_ascii)

    def test_get_source_file(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fodder')
        encoding = inspection.get_encoding_file(os.path.join(path, 'encoding_ascii.py'))
        self.assertEqual(encoding, 'ascii')
        encoding = inspection.get_encoding_file(os.path.join(path, 'encoding_latin1.py'))
        self.assertEqual(encoding, 'latin1')
        encoding = inspection.get_encoding_file(os.path.join(path, 'encoding_utf8.py'))
        self.assertEqual(encoding, 'utf-8')

    @unittest.skipIf(pypy, "pypy builtin signatures aren't complete")
    def test_getfuncprops_print(self):
        props = inspection.getfuncprops('print', print)
        self.assertEqual(props.func, 'print')
        self.assertIn('end', props.argspec.kwonly)
        self.assertIn('file', props.argspec.kwonly)
        self.assertIn('flush', props.argspec.kwonly)
        self.assertIn('sep', props.argspec.kwonly)
        if _is_py311:
            self.assertEqual(repr(props.argspec.kwonly_defaults['file']), 'None')
        else:
            self.assertEqual(repr(props.argspec.kwonly_defaults['file']), 'sys.stdout')
        self.assertEqual(repr(props.argspec.kwonly_defaults['flush']), 'False')

    @unittest.skipUnless(numpy is not None and numpy.__version__ >= '1.18', 'requires numpy >= 1.18')
    def test_getfuncprops_numpy_array(self):
        props = inspection.getfuncprops('array', numpy.array)
        self.assertEqual(props.func, 'array')
        self.assertEqual(props.argspec.args, ['object', 'dtype'])

    def test_issue_966_freestanding(self):

        def fun(number, lst=[]):
            """
            Return a list of numbers

            Example:
            ========
            C.cmethod(1337, [1, 2]) # => [1, 2, 1337]
            """
            return lst + [number]

        def fun_annotations(number: int, lst: List[int]=[]) -> List[int]:
            """
            Return a list of numbers

            Example:
            ========
            C.cmethod(1337, [1, 2]) # => [1, 2, 1337]
            """
            return lst + [number]
        props = inspection.getfuncprops('fun', fun)
        self.assertEqual(props.func, 'fun')
        self.assertEqual(props.argspec.args, ['number', 'lst'])
        self.assertEqual(repr(props.argspec.defaults[0]), '[]')
        props = inspection.getfuncprops('fun_annotations', fun_annotations)
        self.assertEqual(props.func, 'fun_annotations')
        self.assertEqual(props.argspec.args, ['number', 'lst'])
        self.assertEqual(repr(props.argspec.defaults[0]), '[]')

    def test_issue_966_class_method(self):

        class Issue966(Sequence):

            @classmethod
            def cmethod(cls, number: int, lst: List[int]=[]):
                """
                Return a list of numbers

                Example:
                ========
                C.cmethod(1337, [1, 2]) # => [1, 2, 1337]
                """
                return lst + [number]

            @classmethod
            def bmethod(cls, number, lst):
                """
                Return a list of numbers

                Example:
                ========
                C.cmethod(1337, [1, 2]) # => [1, 2, 1337]
                """
                return lst + [number]
        props = inspection.getfuncprops('bmethod', inspection.getattr_safe(Issue966, 'bmethod'))
        self.assertEqual(props.func, 'bmethod')
        self.assertEqual(props.argspec.args, ['number', 'lst'])
        props = inspection.getfuncprops('cmethod', inspection.getattr_safe(Issue966, 'cmethod'))
        self.assertEqual(props.func, 'cmethod')
        self.assertEqual(props.argspec.args, ['number', 'lst'])
        self.assertEqual(repr(props.argspec.defaults[0]), '[]')

    def test_issue_966_static_method(self):

        class Issue966(Sequence):

            @staticmethod
            def cmethod(number: int, lst: List[int]=[]):
                """
                Return a list of numbers

                Example:
                ========
                C.cmethod(1337, [1, 2]) # => [1, 2, 1337]
                """
                return lst + [number]

            @staticmethod
            def bmethod(number, lst):
                """
                Return a list of numbers

                Example:
                ========
                C.cmethod(1337, [1, 2]) # => [1, 2, 1337]
                """
                return lst + [number]
        props = inspection.getfuncprops('bmethod', inspection.getattr_safe(Issue966, 'bmethod'))
        self.assertEqual(props.func, 'bmethod')
        self.assertEqual(props.argspec.args, ['number', 'lst'])
        props = inspection.getfuncprops('cmethod', inspection.getattr_safe(Issue966, 'cmethod'))
        self.assertEqual(props.func, 'cmethod')
        self.assertEqual(props.argspec.args, ['number', 'lst'])
        self.assertEqual(repr(props.argspec.defaults[0]), '[]')