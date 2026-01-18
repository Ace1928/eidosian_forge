from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
class TestArgcompleteREPL(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def run_completer(self, parser, completer, command, point=None, **kwargs):
        cword_prequote, cword_prefix, cword_suffix, comp_words, first_colon_pos = split_line(command)
        completions = completer._get_completions(comp_words, cword_prefix, cword_prequote, first_colon_pos)
        return completions

    def test_repl_multiple_complete(self):
        p = ArgumentParser()
        p.add_argument('--foo')
        p.add_argument('--bar')
        c = CompletionFinder(p, always_complete_options=True)
        completions = self.run_completer(p, c, 'prog ')
        assert set(completions) == set(['-h', '--help', '--foo', '--bar'])
        completions = self.run_completer(p, c, 'prog --')
        assert set(completions) == set(['--help', '--foo', '--bar'])

    def test_repl_parse_after_complete(self):
        p = ArgumentParser()
        p.add_argument('--foo', required=True)
        p.add_argument('bar', choices=['bar'])
        c = CompletionFinder(p, always_complete_options=True)
        completions = self.run_completer(p, c, 'prog ')
        assert set(completions) == set(['-h', '--help', '--foo', 'bar'])
        args = p.parse_args(['--foo', 'spam', 'bar'])
        assert args.foo == 'spam'
        assert args.bar == 'bar'
        with self.assertRaises(SystemExit):
            p.parse_args(['--foo', 'spam'])
        with self.assertRaises(SystemExit):
            p.parse_args(['bar'])

    def test_repl_subparser_parse_after_complete(self):
        p = ArgumentParser()
        sp = p.add_subparsers().add_parser('foo')
        sp.add_argument('bar', choices=['bar'])
        c = CompletionFinder(p, always_complete_options=True)
        completions = self.run_completer(p, c, 'prog foo ')
        assert set(completions) == set(['-h', '--help', 'bar'])
        args = p.parse_args(['foo', 'bar'])
        assert args.bar == 'bar'
        with self.assertRaises(SystemExit):
            p.parse_args(['foo'])

    def test_repl_subcommand(self):
        p = ArgumentParser()
        p.add_argument('--foo')
        p.add_argument('--bar')
        s = p.add_subparsers()
        s.add_parser('list')
        s.add_parser('set')
        show = s.add_parser('show')

        def abc():
            pass
        show.add_argument('--test')
        ss = show.add_subparsers()
        de = ss.add_parser('depth')
        de.set_defaults(func=abc)
        c = CompletionFinder(p, always_complete_options=True)
        expected_outputs = (('prog ', ['-h', '--help', '--foo', '--bar', 'list', 'show', 'set']), ('prog li', ['list ']), ('prog s', ['show', 'set']), ('prog show ', ['--test', 'depth', '-h', '--help']), ('prog show d', ['depth ']), ('prog show depth ', ['-h', '--help']))
        for cmd, output in expected_outputs:
            self.assertEqual(set(self.run_completer(p, c, cmd)), set(output))

    def test_repl_reuse_parser_with_positional(self):
        p = ArgumentParser()
        p.add_argument('foo', choices=['aa', 'bb', 'cc'])
        p.add_argument('bar', choices=['d', 'e'])
        c = CompletionFinder(p, always_complete_options=True)
        self.assertEqual(set(self.run_completer(p, c, 'prog ')), set(['-h', '--help', 'aa', 'bb', 'cc']))
        self.assertEqual(set(self.run_completer(p, c, 'prog aa ')), set(['-h', '--help', 'd', 'e']))
        self.assertEqual(set(self.run_completer(p, c, 'prog ')), set(['-h', '--help', 'aa', 'bb', 'cc']))