import os
import sys
import threading
from io import BytesIO
from textwrap import dedent
import configobj
from testtools import matchers
from .. import (bedding, branch, config, controldir, diff, errors, lock,
from .. import registry as _mod_registry
from .. import tests, trace
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..bzr import remote
from ..transport import remote as transport_remote
from . import features, scenarios, test_server
class TestLocationSection(tests.TestCase):

    def get_section(self, options, extra_path):
        section = config.Section('foo', options)
        return config.LocationSection(section, extra_path)

    def test_simple_option(self):
        section = self.get_section({'foo': 'bar'}, '')
        self.assertEqual('bar', section.get('foo'))

    def test_option_with_extra_path(self):
        section = self.get_section({'foo': 'bar', 'foo:policy': 'appendpath'}, 'baz')
        self.assertEqual('bar/baz', section.get('foo'))

    def test_invalid_policy(self):
        section = self.get_section({'foo': 'bar', 'foo:policy': 'die'}, 'baz')
        self.assertEqual('bar', section.get('foo'))