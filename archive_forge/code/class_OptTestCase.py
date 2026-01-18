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
class OptTestCase(base.BaseTestCase):

    def test_opt_eq(self):
        d1 = cfg.ListOpt('oldfoo')
        d2 = cfg.ListOpt('oldfoo')
        self.assertEqual(d1, d2)

    def test_opt_not_eq(self):
        d1 = cfg.ListOpt('oldfoo')
        d2 = cfg.ListOpt('oldbar')
        self.assertNotEqual(d1, d2)

    def test_illegal_name(self):
        self.assertRaises(ValueError, cfg.BoolOpt, '_foo')