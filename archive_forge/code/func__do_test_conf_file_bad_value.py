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
def _do_test_conf_file_bad_value(self, opt_class):
    self.conf.register_opt(opt_class('foo'))
    paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = bar\n')])
    self.conf(['--config-file', paths[0]])
    self.assertRaises(ValueError, getattr, self.conf, 'foo')
    self.assertRaises(cfg.ConfigFileValueError, self.conf._get, 'foo')