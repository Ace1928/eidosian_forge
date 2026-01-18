from unittest import mock
from oslo_serialization import jsonutils
import sys
from keystoneauth1 import fixture
import requests
class FakeApp(object):

    def __init__(self, _stdout, _log):
        self.stdout = _stdout
        self.client_manager = None
        self.stdin = sys.stdin
        self.stdout = _stdout or sys.stdout
        self.stderr = sys.stderr
        self.log = _log