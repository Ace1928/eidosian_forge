from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
class TestSplitLine(unittest.TestCase):

    def setUp(self):
        self._os_environ = os.environ
        os.environ = os.environ.copy()
        os.environ['_ARGCOMPLETE_COMP_WORDBREAKS'] = COMP_WORDBREAKS

    def tearDown(self):
        os.environ = self._os_environ

    def prefix(self, line):
        return split_line(line)[1]

    def wordbreak(self, line):
        return split_line(line)[4]

    def test_simple(self):
        self.assertEqual(self.prefix('a b c'), 'c')

    def test_escaped_special(self):
        self.assertEqual(self.prefix('a\\$b'), 'a$b')
        self.assertEqual(self.prefix('a\\`b'), 'a`b')

    def test_unescaped_special(self):
        self.assertEqual(self.prefix('a$b'), 'a$b')
        self.assertEqual(self.prefix('a`b'), 'a`b')

    @unittest.expectedFailure
    def test_escaped_special_in_double_quotes(self):
        self.assertEqual(self.prefix('"a\\$b'), 'a$b')
        self.assertEqual(self.prefix('"a\\`b'), 'a`b')

    def test_punctuation(self):
        self.assertEqual(self.prefix('a,'), 'a,')

    def test_last_wordbreak_pos(self):
        self.assertEqual(self.wordbreak('a'), None)
        self.assertEqual(self.wordbreak('a b:c'), 1)
        self.assertEqual(self.wordbreak('a b:c=d'), 3)
        self.assertEqual(self.wordbreak('a b:c=d '), None)
        self.assertEqual(self.wordbreak('a b:c=d e'), None)
        self.assertEqual(self.wordbreak('"b:c'), None)
        self.assertEqual(self.wordbreak('"b:c=d'), None)
        self.assertEqual(self.wordbreak('"b:c=d"'), None)
        self.assertEqual(self.wordbreak('"b:c=d" '), None)