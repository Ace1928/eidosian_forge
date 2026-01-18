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
class TildeExpansionTestCase(BaseTestCase):

    def test_config_file_tilde(self):
        homedir = os.path.expanduser('~')
        tmpfile = tempfile.mktemp(dir=homedir, prefix='cfg-', suffix='.conf')
        tmpbase = os.path.basename(tmpfile)
        try:
            self.conf(['--config-file', os.path.join('~', tmpbase)])
        except cfg.ConfigFilesNotFoundError as cfnfe:
            self.assertIn(homedir, str(cfnfe))
        self.useFixture(fixtures.MonkeyPatch('os.path.exists', lambda p: p == tmpfile))
        self.assertEqual(tmpfile, self.conf.find_file(tmpbase))

    def test_config_dir_tilde(self):
        homedir = os.path.expanduser('~')
        try:
            tmpdir = tempfile.mkdtemp(dir=homedir, prefix='cfg-', suffix='.d')
            tmpfile = os.path.join(tmpdir, 'foo.conf')
            self.useFixture(fixtures.MonkeyPatch('glob.glob', lambda p: [tmpfile]))
            e = self.assertRaises(cfg.ConfigFilesNotFoundError, self.conf, ['--config-dir', os.path.join('~', os.path.basename(tmpdir))])
            self.assertIn(tmpdir, str(e))
        finally:
            try:
                shutil.rmtree(tmpdir)
            except OSError as exc:
                if exc.errno != 2:
                    raise