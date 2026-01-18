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
class TestStackExpandOptions(tests.TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self.overrideAttr(config, 'option_registry', config.OptionRegistry())
        self.registry = config.option_registry
        store = config.TransportIniFileStore(self.get_transport(), 'foo.conf')
        self.conf = config.Stack([store.get_sections], store)

    def assertExpansion(self, expected, string, env=None):
        self.assertEqual(expected, self.conf.expand_options(string, env))

    def test_no_expansion(self):
        self.assertExpansion('foo', 'foo')

    def test_expand_default_value(self):
        self.conf.store._load_from_string(b'bar=baz')
        self.registry.register(config.Option('foo', default='{bar}'))
        self.assertEqual('baz', self.conf.get('foo', expand=True))

    def test_expand_default_from_env(self):
        self.conf.store._load_from_string(b'bar=baz')
        self.registry.register(config.Option('foo', default_from_env=['FOO']))
        self.overrideEnv('FOO', '{bar}')
        self.assertEqual('baz', self.conf.get('foo', expand=True))

    def test_expand_default_on_failed_conversion(self):
        self.conf.store._load_from_string(b'baz=bogus\nbar=42\nfoo={baz}')
        self.registry.register(config.Option('foo', default='{bar}', from_unicode=config.int_from_store))
        self.assertEqual(42, self.conf.get('foo', expand=True))

    def test_env_adding_options(self):
        self.assertExpansion('bar', '{foo}', {'foo': 'bar'})

    def test_env_overriding_options(self):
        self.conf.store._load_from_string(b'foo=baz')
        self.assertExpansion('bar', '{foo}', {'foo': 'bar'})

    def test_simple_ref(self):
        self.conf.store._load_from_string(b'foo=xxx')
        self.assertExpansion('xxx', '{foo}')

    def test_unknown_ref(self):
        self.assertRaises(config.ExpandingUnknownOption, self.conf.expand_options, '{foo}')

    def test_illegal_def_is_ignored(self):
        self.assertExpansion('{1,2}', '{1,2}')
        self.assertExpansion('{ }', '{ }')
        self.assertExpansion('${Foo,f}', '${Foo,f}')

    def test_indirect_ref(self):
        self.conf.store._load_from_string(b'\nfoo=xxx\nbar={foo}\n')
        self.assertExpansion('xxx', '{bar}')

    def test_embedded_ref(self):
        self.conf.store._load_from_string(b'\nfoo=xxx\nbar=foo\n')
        self.assertExpansion('xxx', '{{bar}}')

    def test_simple_loop(self):
        self.conf.store._load_from_string(b'foo={foo}')
        self.assertRaises(config.OptionExpansionLoop, self.conf.expand_options, '{foo}')

    def test_indirect_loop(self):
        self.conf.store._load_from_string(b'\nfoo={bar}\nbar={baz}\nbaz={foo}')
        e = self.assertRaises(config.OptionExpansionLoop, self.conf.expand_options, '{foo}')
        self.assertEqual('foo->bar->baz', e.refs)
        self.assertEqual('{foo}', e.string)

    def test_list(self):
        self.conf.store._load_from_string(b'\nfoo=start\nbar=middle\nbaz=end\nlist={foo},{bar},{baz}\n')
        self.registry.register(config.ListOption('list'))
        self.assertEqual(['start', 'middle', 'end'], self.conf.get('list', expand=True))

    def test_cascading_list(self):
        self.conf.store._load_from_string(b'\nfoo=start,{bar}\nbar=middle,{baz}\nbaz=end\nlist={foo}\n')
        self.registry.register(config.ListOption('list'))
        self.registry.register(config.ListOption('baz'))
        self.assertEqual(['start', 'middle', 'end'], self.conf.get('list', expand=True))

    def test_pathologically_hidden_list(self):
        self.conf.store._load_from_string(b'\nfoo=bin\nbar=go\nstart={foo\nmiddle=},{\nend=bar}\nhidden={start}{middle}{end}\n')
        self.registry.register(config.ListOption('hidden'))
        self.assertEqual(['bin', 'go'], self.conf.get('hidden', expand=True))