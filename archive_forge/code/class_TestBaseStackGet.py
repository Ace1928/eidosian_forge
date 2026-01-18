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
class TestBaseStackGet(tests.TestCase):

    def setUp(self):
        super().setUp()
        self.overrideAttr(config, 'option_registry', config.OptionRegistry())

    def test_get_first_definition(self):
        store1 = config.IniFileStore()
        store1._load_from_string(b'foo=bar')
        store2 = config.IniFileStore()
        store2._load_from_string(b'foo=baz')
        conf = config.Stack([store1.get_sections, store2.get_sections])
        self.assertEqual('bar', conf.get('foo'))

    def test_get_with_registered_default_value(self):
        config.option_registry.register(config.Option('foo', default='bar'))
        conf_stack = config.Stack([])
        self.assertEqual('bar', conf_stack.get('foo'))

    def test_get_without_registered_default_value(self):
        config.option_registry.register(config.Option('foo'))
        conf_stack = config.Stack([])
        self.assertEqual(None, conf_stack.get('foo'))

    def test_get_without_default_value_for_not_registered(self):
        conf_stack = config.Stack([])
        self.assertEqual(None, conf_stack.get('foo'))

    def test_get_for_empty_section_callable(self):
        conf_stack = config.Stack([lambda: []])
        self.assertEqual(None, conf_stack.get('foo'))

    def test_get_for_broken_callable(self):
        conf_stack = config.Stack([object])
        self.assertRaises(TypeError, conf_stack.get, 'foo')