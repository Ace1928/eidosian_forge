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
class ReRegisterOptTestCase(BaseTestCase):

    def test_conf_file_re_register_opt(self):
        opt = cfg.StrOpt('foo')
        self.assertTrue(self.conf.register_opt(opt))
        self.assertFalse(self.conf.register_opt(opt))

    def test_conf_file_re_register_opt_in_group(self):
        group = cfg.OptGroup('blaa')
        self.conf.register_group(group)
        self.conf.register_group(group)
        opt = cfg.StrOpt('foo')
        self.assertTrue(self.conf.register_opt(opt, group=group))
        self.assertFalse(self.conf.register_opt(opt, group='blaa'))