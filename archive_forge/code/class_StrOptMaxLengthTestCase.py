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
class StrOptMaxLengthTestCase(BaseTestCase):

    def test_stropt_max_length_good(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo', max_length=5))
        self.conf(['--foo', '12345'])
        self.assertEqual('12345', self.conf.foo)

    def test_stropt_max_length_bad(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo', max_length=5))
        self.assertRaises(SystemExit, self.conf, ['--foo', '123456'])