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
def _do_dname_test_use(self, opt_class, value, result):
    self._do_deprecated_test(opt_class, value, result, 'oldfoo', dname='oldfoo')