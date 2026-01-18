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
class TestStackGetWithConverter(tests.TestCase):

    def setUp(self):
        super().setUp()
        self.overrideAttr(config, 'option_registry', config.OptionRegistry())
        self.registry = config.option_registry

    def get_conf(self, content=None):
        return config.MemoryStack(content)

    def register_bool_option(self, name, default=None, default_from_env=None):
        b = config.Option(name, help='A boolean.', default=default, default_from_env=default_from_env, from_unicode=config.bool_from_store)
        self.registry.register(b)

    def test_get_default_bool_None(self):
        self.register_bool_option('foo')
        conf = self.get_conf(b'')
        self.assertEqual(None, conf.get('foo'))

    def test_get_default_bool_True(self):
        self.register_bool_option('foo', 'True')
        conf = self.get_conf(b'')
        self.assertEqual(True, conf.get('foo'))

    def test_get_default_bool_False(self):
        self.register_bool_option('foo', False)
        conf = self.get_conf(b'')
        self.assertEqual(False, conf.get('foo'))

    def test_get_default_bool_False_as_string(self):
        self.register_bool_option('foo', 'False')
        conf = self.get_conf(b'')
        self.assertEqual(False, conf.get('foo'))

    def test_get_default_bool_from_env_converted(self):
        self.register_bool_option('foo', 'True', default_from_env=['FOO'])
        self.overrideEnv('FOO', 'False')
        conf = self.get_conf(b'')
        self.assertEqual(False, conf.get('foo'))

    def test_get_default_bool_when_conversion_fails(self):
        self.register_bool_option('foo', default='True')
        conf = self.get_conf(b'foo=invalid boolean')
        self.assertEqual(True, conf.get('foo'))

    def register_integer_option(self, name, default=None, default_from_env=None):
        i = config.Option(name, help='An integer.', default=default, default_from_env=default_from_env, from_unicode=config.int_from_store)
        self.registry.register(i)

    def test_get_default_integer_None(self):
        self.register_integer_option('foo')
        conf = self.get_conf(b'')
        self.assertEqual(None, conf.get('foo'))

    def test_get_default_integer(self):
        self.register_integer_option('foo', 42)
        conf = self.get_conf(b'')
        self.assertEqual(42, conf.get('foo'))

    def test_get_default_integer_as_string(self):
        self.register_integer_option('foo', '42')
        conf = self.get_conf(b'')
        self.assertEqual(42, conf.get('foo'))

    def test_get_default_integer_from_env(self):
        self.register_integer_option('foo', default_from_env=['FOO'])
        self.overrideEnv('FOO', '18')
        conf = self.get_conf(b'')
        self.assertEqual(18, conf.get('foo'))

    def test_get_default_integer_when_conversion_fails(self):
        self.register_integer_option('foo', default='12')
        conf = self.get_conf(b'foo=invalid integer')
        self.assertEqual(12, conf.get('foo'))

    def register_list_option(self, name, default=None, default_from_env=None):
        l = config.ListOption(name, help='A list.', default=default, default_from_env=default_from_env)
        self.registry.register(l)

    def test_get_default_list_None(self):
        self.register_list_option('foo')
        conf = self.get_conf(b'')
        self.assertEqual(None, conf.get('foo'))

    def test_get_default_list_empty(self):
        self.register_list_option('foo', '')
        conf = self.get_conf(b'')
        self.assertEqual([], conf.get('foo'))

    def test_get_default_list_from_env(self):
        self.register_list_option('foo', default_from_env=['FOO'])
        self.overrideEnv('FOO', '')
        conf = self.get_conf(b'')
        self.assertEqual([], conf.get('foo'))

    def test_get_with_list_converter_no_item(self):
        self.register_list_option('foo', None)
        conf = self.get_conf(b'foo=,')
        self.assertEqual([], conf.get('foo'))

    def test_get_with_list_converter_many_items(self):
        self.register_list_option('foo', None)
        conf = self.get_conf(b'foo=m,o,r,e')
        self.assertEqual(['m', 'o', 'r', 'e'], conf.get('foo'))

    def test_get_with_list_converter_embedded_spaces_many_items(self):
        self.register_list_option('foo', None)
        conf = self.get_conf(b'foo=" bar", "baz "')
        self.assertEqual([' bar', 'baz '], conf.get('foo'))

    def test_get_with_list_converter_stripped_spaces_many_items(self):
        self.register_list_option('foo', None)
        conf = self.get_conf(b'foo= bar ,  baz ')
        self.assertEqual(['bar', 'baz'], conf.get('foo'))