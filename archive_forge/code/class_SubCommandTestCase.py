import argparse
import errno
import functools
import io
import logging
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock
import fixtures
from oslotest import base
import testscenarios
from oslo_config import cfg
from oslo_config import types
class SubCommandTestCase(BaseTestCase):

    def test_sub_command(self):

        def add_parsers(subparsers):
            sub = subparsers.add_parser('a')
            sub.add_argument('bar', type=int)
        self.conf.register_cli_opt(cfg.SubCommandOpt('cmd', handler=add_parsers))
        self.assertTrue(hasattr(self.conf, 'cmd'))
        self.conf(['a', '10'])
        self.assertTrue(hasattr(self.conf.cmd, 'name'))
        self.assertTrue(hasattr(self.conf.cmd, 'bar'))
        self.assertEqual('a', self.conf.cmd.name)
        self.assertEqual(10, self.conf.cmd.bar)

    def test_sub_command_with_parent(self):

        def add_parsers(subparsers):
            parent = argparse.ArgumentParser(add_help=False)
            parent.add_argument('bar', type=int)
            subparsers.add_parser('a', parents=[parent])
        self.conf.register_cli_opt(cfg.SubCommandOpt('cmd', handler=add_parsers))
        self.assertTrue(hasattr(self.conf, 'cmd'))
        self.conf(['a', '10'])
        self.assertTrue(hasattr(self.conf.cmd, 'name'))
        self.assertTrue(hasattr(self.conf.cmd, 'bar'))
        self.assertEqual('a', self.conf.cmd.name)
        self.assertEqual(10, self.conf.cmd.bar)

    def test_sub_command_with_dest(self):

        def add_parsers(subparsers):
            subparsers.add_parser('a')
        self.conf.register_cli_opt(cfg.SubCommandOpt('cmd', dest='command', handler=add_parsers))
        self.assertTrue(hasattr(self.conf, 'command'))
        self.conf(['a'])
        self.assertEqual('a', self.conf.command.name)

    def test_sub_command_with_group(self):

        def add_parsers(subparsers):
            sub = subparsers.add_parser('a')
            sub.add_argument('--bar', choices='XYZ')
        self.conf.register_cli_opt(cfg.SubCommandOpt('cmd', handler=add_parsers), group='blaa')
        self.assertTrue(hasattr(self.conf, 'blaa'))
        self.assertTrue(hasattr(self.conf.blaa, 'cmd'))
        self.conf(['a', '--bar', 'Z'])
        self.assertTrue(hasattr(self.conf.blaa.cmd, 'name'))
        self.assertTrue(hasattr(self.conf.blaa.cmd, 'bar'))
        self.assertEqual('a', self.conf.blaa.cmd.name)
        self.assertEqual('Z', self.conf.blaa.cmd.bar)

    def test_sub_command_not_cli(self):
        self.conf.register_opt(cfg.SubCommandOpt('cmd'))
        self.conf([])

    def test_sub_command_resparse(self):

        def add_parsers(subparsers):
            subparsers.add_parser('a')
        self.conf.register_cli_opt(cfg.SubCommandOpt('cmd', handler=add_parsers))
        foo_opt = cfg.StrOpt('foo')
        self.conf.register_cli_opt(foo_opt)
        self.conf(['--foo=bar', 'a'])
        self.assertTrue(hasattr(self.conf.cmd, 'name'))
        self.assertEqual('a', self.conf.cmd.name)
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual('bar', self.conf.foo)
        self.conf.clear()
        self.conf.unregister_opt(foo_opt)
        self.conf(['a'])
        self.assertTrue(hasattr(self.conf.cmd, 'name'))
        self.assertEqual('a', self.conf.cmd.name)
        self.assertFalse(hasattr(self.conf, 'foo'))

    def test_sub_command_no_handler(self):
        self.conf.register_cli_opt(cfg.SubCommandOpt('cmd'))
        self.useFixture(fixtures.MonkeyPatch('sys.stderr', io.StringIO()))
        self.assertRaises(SystemExit, self.conf, [])
        self.assertIn('error', sys.stderr.getvalue())

    def test_sub_command_with_help(self):

        def add_parsers(subparsers):
            subparsers.add_parser('a')
        self.conf.register_cli_opt(cfg.SubCommandOpt('cmd', title='foo foo', description='bar bar', help='blaa blaa', handler=add_parsers))
        self.useFixture(fixtures.MonkeyPatch('sys.stdout', io.StringIO()))
        self.assertRaises(SystemExit, self.conf, ['--help'])
        self.assertIn('foo foo', sys.stdout.getvalue())
        self.assertIn('bar bar', sys.stdout.getvalue())
        self.assertIn('blaa blaa', sys.stdout.getvalue())

    def test_sub_command_errors(self):

        def add_parsers(subparsers):
            sub = subparsers.add_parser('a')
            sub.add_argument('--bar')
        self.conf.register_cli_opt(cfg.BoolOpt('bar'))
        self.conf.register_cli_opt(cfg.SubCommandOpt('cmd', handler=add_parsers))
        self.conf(['a'])
        self.assertRaises(cfg.DuplicateOptError, getattr, self.conf.cmd, 'bar')
        self.assertRaises(cfg.NoSuchOptError, getattr, self.conf.cmd, 'foo')

    def test_sub_command_multiple(self):
        self.conf.register_cli_opt(cfg.SubCommandOpt('cmd1'))
        self.conf.register_cli_opt(cfg.SubCommandOpt('cmd2'))
        self.useFixture(fixtures.MonkeyPatch('sys.stderr', io.StringIO()))
        self.assertRaises(SystemExit, self.conf, [])
        self.assertIn('multiple', sys.stderr.getvalue())