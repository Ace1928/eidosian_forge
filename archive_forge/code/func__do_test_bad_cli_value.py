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
def _do_test_bad_cli_value(self, opt_class):
    self.conf.register_cli_opt(opt_class('foo'))
    self.useFixture(fixtures.MonkeyPatch('sys.stderr', io.StringIO()))
    self.assertRaises(SystemExit, self.conf, ['--foo', 'bar'])
    self.assertIn('foo', sys.stderr.getvalue())
    self.assertIn('bar', sys.stderr.getvalue())