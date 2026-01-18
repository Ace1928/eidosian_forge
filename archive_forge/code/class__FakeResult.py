import io
import sys
from unittest import mock
from urllib import parse
from novaclient import base
from novaclient import exceptions
from novaclient.tests.unit import fakes
from novaclient.tests.unit import utils as test_utils
from novaclient import utils
from novaclient.v2 import servers
class _FakeResult(object):

    def __init__(self, name, value):
        self.name = name
        self.value = value