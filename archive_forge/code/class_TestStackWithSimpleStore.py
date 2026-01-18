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
class TestStackWithSimpleStore(tests.TestCase):

    def setUp(self):
        super().setUp()
        self.overrideAttr(config, 'option_registry', config.OptionRegistry())
        self.registry = config.option_registry

    def get_conf(self, content=None):
        return config.MemoryStack(content)

    def test_override_value_from_env(self):
        self.overrideEnv('FOO', None)
        self.registry.register(config.Option('foo', default='bar', override_from_env=['FOO']))
        self.overrideEnv('FOO', 'quux')
        conf = self.get_conf(b'foo=store')
        self.assertEqual('quux', conf.get('foo'))

    def test_first_override_value_from_env_wins(self):
        self.overrideEnv('NO_VALUE', None)
        self.overrideEnv('FOO', None)
        self.overrideEnv('BAZ', None)
        self.registry.register(config.Option('foo', default='bar', override_from_env=['NO_VALUE', 'FOO', 'BAZ']))
        self.overrideEnv('FOO', 'foo')
        self.overrideEnv('BAZ', 'baz')
        conf = self.get_conf(b'foo=store')
        self.assertEqual('foo', conf.get('foo'))