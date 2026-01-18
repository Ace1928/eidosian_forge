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
def _do_test_log_opt_values(self, args):
    self.conf.register_cli_opt(cfg.StrOpt('foo'))
    self.conf.register_cli_opt(cfg.StrOpt('passwd', secret=True))
    self.conf.register_group(cfg.OptGroup('blaa'))
    self.conf.register_cli_opt(cfg.StrOpt('bar'), 'blaa')
    self.conf.register_cli_opt(cfg.StrOpt('key', secret=True), 'blaa')
    self.conf(args)
    logger = self.FakeLogger(self, 666)
    self.conf.log_opt_values(logger, 666)
    self.assertEqual(['*' * 80, 'Configuration options gathered from:', "command line args: ['--foo', 'this', '--blaa-bar', 'that', '--blaa-key', 'admin', '--passwd', 'hush']", 'config files: []', '=' * 80, 'config_dir                     = []', 'config_file                    = []', 'config_source                  = []', 'foo                            = this', 'passwd                         = ****', 'blaa.bar                       = that', 'blaa.key                       = ****', '*' * 80], logger.logged)