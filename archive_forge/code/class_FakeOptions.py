from unittest import mock
from oslo_serialization import jsonutils
import sys
from keystoneauth1 import fixture
import requests
class FakeOptions(object):

    def __init__(self, **kwargs):
        self.os_beta_command = False