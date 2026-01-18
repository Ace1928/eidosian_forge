from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
class _TestSh(object):
    """
    Contains tests which should work in any shell using argcomplete.

    Tests use the test program in this directory named ``prog``.
    All commands are expected to input one of the valid choices
    which is then printed and collected by the shell wrapper.

    Any tabs in the input line simulate the user pressing tab.
    For example, ``self.sh.run_command('prog basic "b	r	')`` will
    simulate having the user:

    1. Type ``prog basic "b``
    2. Push tab, which returns ``['bar', 'baz']``, filling in ``a``
    3. Type ``r``
    4. Push tab, which returns ``['bar']``, filling in ``" ``
    5. Push enter, submitting ``prog basic "bar" ``

    The end result should be ``bar`` being printed to the screen.
    """
    sh = None
    expected_failures = []

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        for name in cls.expected_failures:
            test = getattr(cls, name)

            @unittest.expectedFailure
            def wrapped(self, test=test):
                test(self)
            setattr(cls, name, wrapped)
        super(_TestSh, cls).setUpClass(*args, **kwargs)

    def setUp(self):
        raise NotImplementedError

    def tearDown(self):
        with self.assertRaises(pexpect.EOF):
            self.sh.run_command('exit')

    def test_simple_completion(self):
        self.assertEqual(self.sh.run_command('prog basic f\t'), 'foo\r\n')

    def test_partial_completion(self):
        self.assertEqual(self.sh.run_command('prog basic b\tr'), 'bar\r\n')

    def test_single_quoted_completion(self):
        self.assertEqual(self.sh.run_command("prog basic 'f\t"), 'foo\r\n')

    def test_double_quoted_completion(self):
        self.assertEqual(self.sh.run_command('prog basic "f\t'), 'foo\r\n')

    def test_unquoted_space(self):
        self.assertEqual(self.sh.run_command('prog space f\t'), 'foo bar\r\n')

    def test_quoted_space(self):
        self.assertEqual(self.sh.run_command('prog space "f\t'), 'foo bar\r\n')

    def test_continuation(self):
        self.assertEqual(self.sh.run_command('prog basic f\t--'), 'foo\r\n')
        self.assertEqual(self.sh.run_command('prog cont f\t--'), 'foo=--\r\n')
        self.assertEqual(self.sh.run_command('prog cont bar\t--'), 'bar/--\r\n')
        self.assertEqual(self.sh.run_command('prog cont baz\t--'), 'baz:--\r\n')

    def test_quoted_exact(self):
        self.assertEqual(self.sh.run_command('prog basic "f\t--'), 'foo\r\n')

    def test_special_characters(self):
        self.assertEqual(self.sh.run_command('prog spec d\tf'), 'd$e$f\r\n')
        self.assertEqual(self.sh.run_command('prog spec x\t'), 'x!x\r\n')
        self.assertEqual(self.sh.run_command('prog spec y\t'), 'y\\y\r\n')

    def test_special_characters_single_quoted(self):
        self.assertEqual(self.sh.run_command("prog spec 'd\tf'"), 'd$e$f\r\n')

    def test_special_characters_double_quoted(self):
        self.assertEqual(self.sh.run_command('prog spec "d\tf"'), 'd$e$f\r\n')

    def test_parse_special_characters(self):
        self.assertEqual(self.sh.run_command('prog spec d$e$\tf'), 'd$e$f\r\n')
        self.assertEqual(self.sh.run_command('prog spec d$e\tf'), 'd$e$f\r\n')
        self.assertEqual(self.sh.run_command("prog spec 'd$e\tf\t"), 'd$e$f\r\n')

    def test_parse_special_characters_dollar(self):
        self.assertEqual(self.sh.run_command('prog spec "d$e\tf\t'), 'd$e$f\r\n')

    def test_exclamation_in_double_quotes(self):
        self.assertEqual(self.sh.run_command('prog spec "x\t'), 'x!x\r\n')

    def test_quotes(self):
        self.assertEqual(self.sh.run_command('prog quote 1\t'), "1'1\r\n")
        self.assertEqual(self.sh.run_command('prog quote 2\t'), '2"2\r\n')

    def test_single_quotes_in_double_quotes(self):
        self.assertEqual(self.sh.run_command('prog quote "1\t'), "1'1\r\n")

    def test_single_quotes_in_single_quotes(self):
        self.assertEqual(self.sh.run_command("prog quote '1\t"), "1'1\r\n")

    def test_wordbreak_chars(self):
        self.assertEqual(self.sh.run_command('prog break a\tc'), 'a:b:c\r\n')
        self.assertEqual(self.sh.run_command('prog break a:b:\tc'), 'a:b:c\r\n')
        self.assertEqual(self.sh.run_command('prog break a:b\tc'), 'a:b:c\r\n')
        self.assertEqual(self.sh.run_command("prog break 'a\tc'"), 'a:b:c\r\n')
        self.assertEqual(self.sh.run_command("prog break 'a:b\tc\t"), 'a:b:c\r\n')
        self.assertEqual(self.sh.run_command('prog break "a\tc"'), 'a:b:c\r\n')
        self.assertEqual(self.sh.run_command('prog break "a:b\tc\t'), 'a:b:c\r\n')

    def test_completion_environment(self):
        self.assertEqual(self.sh.run_command('prog env o\t'), 'ok\r\n')

    def test_comp_point(self):
        self.assertEqual(self.sh.run_command('export POINT=1'), '')
        self.assertEqual(self.sh.run_command('prog point hi\t'), '13\r\n')
        self.assertEqual(self.sh.run_command('prog point hi \t'), '14\r\n')
        self.assertEqual(self.sh.run_command('prog point 你好嘚瑟\t'), '15\r\n')
        self.assertEqual(self.sh.run_command('prog point 你好嘚瑟 \t'), '16\r\n')