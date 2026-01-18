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
def _prep_test_str_sub(self, foo_default=None, bar_default=None):
    self.conf.register_cli_opt(cfg.StrOpt('foo', default=foo_default))
    self.conf.register_cli_opt(cfg.StrOpt('bar', default=bar_default))