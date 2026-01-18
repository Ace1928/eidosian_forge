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
class UnregisterOptTestCase(BaseTestCase):

    def test_unregister_opt(self):
        opts = [cfg.StrOpt('foo'), cfg.StrOpt('bar')]
        self.conf.register_opts(opts)
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertTrue(hasattr(self.conf, 'bar'))
        self.conf.unregister_opt(opts[0])
        self.assertFalse(hasattr(self.conf, 'foo'))
        self.assertTrue(hasattr(self.conf, 'bar'))
        self.conf([])
        self.assertRaises(cfg.ArgsAlreadyParsedError, self.conf.unregister_opt, opts[1])
        self.conf.clear()
        self.assertTrue(hasattr(self.conf, 'bar'))
        self.conf.unregister_opts(opts)

    def test_unregister_opt_from_group(self):
        opt = cfg.StrOpt('foo')
        self.conf.register_opt(opt, group='blaa')
        self.assertTrue(hasattr(self.conf, 'blaa'))
        self.assertTrue(hasattr(self.conf.blaa, 'foo'))
        self.conf.unregister_opt(opt, group='blaa')
        self.assertFalse(hasattr(self.conf.blaa, 'foo'))