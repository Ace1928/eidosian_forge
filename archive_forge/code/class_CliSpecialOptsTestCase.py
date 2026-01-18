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
class CliSpecialOptsTestCase(BaseTestCase):

    def test_help(self):
        self.useFixture(fixtures.MonkeyPatch('sys.stdout', io.StringIO()))
        self.assertRaises(SystemExit, self.conf, ['--help'])
        self.assertIn('usage: test', sys.stdout.getvalue())
        self.assertIn('[--version]', sys.stdout.getvalue())
        self.assertIn('[-h]', sys.stdout.getvalue())
        self.assertIn('--help', sys.stdout.getvalue())
        self.assertIn('[--config-dir DIR]', sys.stdout.getvalue())
        self.assertIn('[--config-file PATH]', sys.stdout.getvalue())

    def test_version(self):
        stream_name = 'stdout'
        self.useFixture(fixtures.MonkeyPatch('sys.%s' % stream_name, io.StringIO()))
        self.assertRaises(SystemExit, self.conf, ['--version'])
        self.assertIn('1.0', getattr(sys, stream_name).getvalue())

    def test_config_file(self):
        paths = self.create_tempfiles([('1', '[DEFAULT]'), ('2', '[DEFAULT]')])
        self.conf(['--config-file', paths[0], '--config-file', paths[1]])
        self.assertEqual(paths, self.conf.config_file)