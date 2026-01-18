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
def _capture_stream(self, stream_name):
    self.useFixture(fixtures.MonkeyPatch('sys.%s' % stream_name, io.StringIO()))
    return getattr(sys, stream_name)