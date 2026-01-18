import os
import unittest
import getpass
import urllib
import warnings
from test.support.warnings_helper import check_warnings
from distutils.command import register as register_module
from distutils.command.register import register
from distutils.errors import DistutilsSetupError
from distutils.log import INFO
from distutils.tests.test_config import BasePyPIRCCommandTestCase
class FakeOpener(object):
    """Fakes a PyPI server"""

    def __init__(self):
        self.reqs = []

    def __call__(self, *args):
        return self

    def open(self, req, data=None, timeout=None):
        self.reqs.append(req)
        return self

    def read(self):
        return b'xxx'

    def getheader(self, name, default=None):
        return {'content-type': 'text/plain; charset=utf-8'}.get(name.lower(), default)