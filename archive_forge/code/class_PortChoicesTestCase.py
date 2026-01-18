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
class PortChoicesTestCase(BaseTestCase):

    def test_choice_default(self):
        self.conf.register_cli_opt(cfg.PortOpt('port', default=455, choices=[80, 455]))
        self.conf([])
        self.assertEqual(455, self.conf.port)

    def test_choice_good_with_list(self):
        self.conf.register_cli_opt(cfg.PortOpt('port', choices=[80, 8080]))
        self.conf(['--port', '80'])
        self.assertEqual(80, self.conf.port)

    def test_choice_good_with_tuple(self):
        self.conf.register_cli_opt(cfg.PortOpt('port', choices=(80, 8080)))
        self.conf(['--port', '80'])
        self.assertEqual(80, self.conf.port)

    def test_choice_bad(self):
        self.conf.register_cli_opt(cfg.PortOpt('port', choices=[80, 8080]))
        self.assertRaises(SystemExit, self.conf, ['--port', '8181'])

    def test_choice_out_range(self):
        self.assertRaisesRegex(ValueError, 'out of bounds', cfg.PortOpt, 'port', choices=[80, 65537, 0])

    def test_conf_file_choice_value(self):
        self.conf.register_opt(cfg.PortOpt('port', choices=[80, 8080]))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nport = 80\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'port'))
        self.assertEqual(80, self.conf.port)

    def test_conf_file_bad_choice_value(self):
        self.conf.register_opt(cfg.PortOpt('port', choices=[80, 8080]))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nport = 8181\n')])
        self.conf(['--config-file', paths[0]])
        self.assertRaises(cfg.ConfigFileValueError, self.conf._get, 'port')
        self.assertRaises(ValueError, getattr, self.conf, 'port')

    def test_conf_file_choice_value_override(self):
        self.conf.register_cli_opt(cfg.PortOpt('port', choices=[80, 8080]))
        paths = self.create_tempfiles([('1', '[DEFAULT]\nport = 80\n'), ('2', '[DEFAULT]\nport = 8080\n')])
        self.conf(['--port', '80', '--config-file', paths[0], '--config-file', paths[1]])
        self.assertTrue(hasattr(self.conf, 'port'))
        self.assertEqual(8080, self.conf.port)

    def test_conf_file_choice_bad_default(self):
        self.assertRaises(cfg.DefaultValueError, cfg.PortOpt, 'port', choices=[80, 8080], default=8181)