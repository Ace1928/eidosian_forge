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
def _do_pos_test(self, opt_class, default, cli_args, value):
    self.conf.register_cli_opt(opt_class('foo', default=default, positional=True, required=False))
    self.conf(cli_args)
    self.assertTrue(hasattr(self.conf, 'foo'))
    self.assertEqual(value, self.conf.foo)