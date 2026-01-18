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
class PrintHelpTestCase(base.BaseTestCase):

    def test_print_help_without_init(self):
        conf = cfg.ConfigOpts()
        conf.register_opts([])
        self.assertRaises(cfg.NotInitializedError, conf.print_help)

    def test_print_help_with_clear(self):
        conf = cfg.ConfigOpts()
        conf.register_opts([])
        conf([])
        conf.clear()
        self.assertRaises(cfg.NotInitializedError, conf.print_help)