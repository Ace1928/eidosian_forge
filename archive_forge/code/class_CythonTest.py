from __future__ import absolute_import
import os
import re
import unittest
import shlex
import sys
import tempfile
import textwrap
from io import open
from functools import partial
from .Compiler import Errors
from .CodeWriter import CodeWriter
from .Compiler.TreeFragment import TreeFragment, strip_common_indent
from .Compiler.Visitor import TreeVisitor, VisitorTransform
from .Compiler import TreePath
class CythonTest(unittest.TestCase):

    def setUp(self):
        Errors.init_thread()

    def tearDown(self):
        Errors.init_thread()

    def assertLines(self, expected, result):
        """Checks that the given strings or lists of strings are equal line by line"""
        if not isinstance(expected, list):
            expected = expected.split(u'\n')
        if not isinstance(result, list):
            result = result.split(u'\n')
        for idx, (expected_line, result_line) in enumerate(zip(expected, result)):
            self.assertEqual(expected_line, result_line, 'Line %d:\nExp: %s\nGot: %s' % (idx, expected_line, result_line))
        self.assertEqual(len(expected), len(result), 'Unmatched lines. Got:\n%s\nExpected:\n%s' % ('\n'.join(expected), u'\n'.join(result)))

    def codeToLines(self, tree):
        writer = CodeWriter()
        writer.write(tree)
        return writer.result.lines

    def codeToString(self, tree):
        return '\n'.join(self.codeToLines(tree))

    def assertCode(self, expected, result_tree):
        result_lines = self.codeToLines(result_tree)
        expected_lines = strip_common_indent(expected.split('\n'))
        for idx, (line, expected_line) in enumerate(zip(result_lines, expected_lines)):
            self.assertEqual(expected_line, line, 'Line %d:\nGot: %s\nExp: %s' % (idx, line, expected_line))
        self.assertEqual(len(result_lines), len(expected_lines), 'Unmatched lines. Got:\n%s\nExpected:\n%s' % ('\n'.join(result_lines), expected))

    def assertNodeExists(self, path, result_tree):
        self.assertNotEqual(TreePath.find_first(result_tree, path), None, "Path '%s' not found in result tree" % path)

    def fragment(self, code, pxds=None, pipeline=None):
        """Simply create a tree fragment using the name of the test-case in parse errors."""
        if pxds is None:
            pxds = {}
        if pipeline is None:
            pipeline = []
        name = self.id()
        if name.startswith('__main__.'):
            name = name[len('__main__.'):]
        name = name.replace('.', '_')
        return TreeFragment(code, name, pxds, pipeline=pipeline)

    def treetypes(self, root):
        return treetypes(root)

    def should_fail(self, func, exc_type=Exception):
        """Calls "func" and fails if it doesn't raise the right exception
        (any exception by default). Also returns the exception in question.
        """
        try:
            func()
            self.fail('Expected an exception of type %r' % exc_type)
        except exc_type as e:
            self.assertTrue(isinstance(e, exc_type))
            return e

    def should_not_fail(self, func):
        """Calls func and succeeds if and only if no exception is raised
        (i.e. converts exception raising into a failed testcase). Returns
        the return value of func."""
        try:
            return func()
        except Exception as exc:
            self.fail(str(exc))