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
def _test_conf_files_mutate(self):
    paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo = old_foo\n[group]\nboo = old_boo\n'), ('2', '[DEFAULT]\nfoo = new_foo\n[group]\nboo = new_boo\n')])
    self.conf(['--config-file', paths[0]])
    shutil.copy(paths[1], paths[0])
    return self.conf.mutate_config_files()