import logging
import sys
import urllib.parse as urlparse
import uuid
import fixtures
from oslo_serialization import jsonutils
import requests
import requests_mock
from requests_mock.contrib import fixture
import testscenarios
import testtools
from keystoneclient.tests.unit import client_fixtures
class NoModuleFinder(object):
    """Disallow further imports of 'module'."""

    def __init__(self, module):
        self.module = module

    def find_module(self, fullname, path):
        if fullname == self.module or fullname.startswith(self.module + '.'):
            raise ImportError