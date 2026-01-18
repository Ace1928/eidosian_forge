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
class MappingInterfaceTestCase(BaseTestCase):

    def test_mapping_interface(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo'))
        self.conf(['--foo', 'bar'])
        self.assertIn('foo', self.conf)
        self.assertIn('config_file', self.conf)
        self.assertEqual(len(self.conf), 4)
        self.assertEqual('bar', self.conf['foo'])
        self.assertEqual('bar', self.conf.get('foo'))
        self.assertIn('bar', list(self.conf.values()))

    def test_mapping_interface_with_group(self):
        self.conf.register_group(cfg.OptGroup('blaa'))
        self.conf.register_cli_opt(cfg.StrOpt('foo'), group='blaa')
        self.conf(['--blaa-foo', 'bar'])
        self.assertIn('blaa', self.conf)
        self.assertIn('foo', list(self.conf['blaa']))
        self.assertEqual(len(self.conf['blaa']), 1)
        self.assertEqual('bar', self.conf['blaa']['foo'])
        self.assertEqual('bar', self.conf['blaa'].get('foo'))
        self.assertIn('bar', self.conf['blaa'].values())
        self.assertEqual(self.conf['blaa'], self.conf.blaa)