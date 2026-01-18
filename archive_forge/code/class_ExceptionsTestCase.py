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
class ExceptionsTestCase(base.BaseTestCase):

    def test_error(self):
        msg = str(cfg.Error('foobar'))
        self.assertEqual('foobar', msg)

    def test_args_already_parsed_error(self):
        msg = str(cfg.ArgsAlreadyParsedError('foobar'))
        self.assertEqual('arguments already parsed: foobar', msg)

    def test_no_such_opt_error(self):
        msg = str(cfg.NoSuchOptError('foo'))
        self.assertEqual('no such option foo in group [DEFAULT]', msg)

    def test_no_such_opt_error_with_group(self):
        msg = str(cfg.NoSuchOptError('foo', cfg.OptGroup('bar')))
        self.assertEqual('no such option foo in group [bar]', msg)

    def test_no_such_group_error(self):
        msg = str(cfg.NoSuchGroupError('bar'))
        self.assertEqual('no such group [bar]', msg)

    def test_duplicate_opt_error(self):
        msg = str(cfg.DuplicateOptError('foo'))
        self.assertEqual('duplicate option: foo', msg)

    def test_required_opt_error(self):
        msg = str(cfg.RequiredOptError('foo'))
        self.assertEqual('value required for option foo in group [DEFAULT]', msg)

    def test_required_opt_error_with_group(self):
        msg = str(cfg.RequiredOptError('foo', cfg.OptGroup('bar')))
        self.assertEqual('value required for option foo in group [bar]', msg)

    def test_template_substitution_error(self):
        msg = str(cfg.TemplateSubstitutionError('foobar'))
        self.assertEqual('template substitution error: foobar', msg)

    def test_config_files_not_found_error(self):
        msg = str(cfg.ConfigFilesNotFoundError(['foo', 'bar']))
        self.assertEqual('Failed to find some config files: foo,bar', msg)

    def test_config_files_permission_denied_error(self):
        msg = str(cfg.ConfigFilesPermissionDeniedError(['foo', 'bar']))
        self.assertEqual('Failed to open some config files: foo,bar', msg)

    def test_config_dir_not_found_error(self):
        msg = str(cfg.ConfigDirNotFoundError('foobar'))
        self.assertEqual('Failed to read config file directory: foobar', msg)

    def test_config_file_parse_error(self):
        msg = str(cfg.ConfigFileParseError('foo', 'foobar'))
        self.assertEqual('Failed to parse foo: foobar', msg)