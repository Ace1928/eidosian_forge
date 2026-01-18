import io
import sys
import textwrap
from unittest import mock
import fixtures
from oslotest import base
import tempfile
import testscenarios
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_config import generator
from oslo_config import types
import yaml
class FakeEP:

    def __init__(self):
        self.name = 'callback_is_expected'
        self.require = self.resolve
        self.load = self.resolve

    def resolve(self, *args, **kwargs):
        raise FakeException()