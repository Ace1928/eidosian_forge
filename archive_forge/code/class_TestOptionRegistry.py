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
class TestOptionRegistry(tests.TestCase):

    def setUp(self):
        super().setUp()
        self.overrideAttr(config, 'option_registry', config.OptionRegistry())
        self.registry = config.option_registry

    def test_register(self):
        opt = config.Option('foo')
        self.registry.register(opt)
        self.assertIs(opt, self.registry.get('foo'))

    def test_registered_help(self):
        opt = config.Option('foo', help='A simple option')
        self.registry.register(opt)
        self.assertEqual('A simple option', self.registry.get_help('foo'))

    def test_dont_register_illegal_name(self):
        self.assertRaises(config.IllegalOptionName, self.registry.register, config.Option(' foo'))
        self.assertRaises(config.IllegalOptionName, self.registry.register, config.Option('bar,'))
    lazy_option = config.Option('lazy_foo', help='Lazy help')

    def test_register_lazy(self):
        self.registry.register_lazy('lazy_foo', self.__module__, 'TestOptionRegistry.lazy_option')
        self.assertIs(self.lazy_option, self.registry.get('lazy_foo'))

    def test_registered_lazy_help(self):
        self.registry.register_lazy('lazy_foo', self.__module__, 'TestOptionRegistry.lazy_option')
        self.assertEqual('Lazy help', self.registry.get_help('lazy_foo'))

    def test_dont_lazy_register_illegal_name(self):
        self.assertRaises(config.IllegalOptionName, self.registry.register_lazy, ' foo', self.__module__, 'TestOptionRegistry.lazy_option')
        self.assertRaises(config.IllegalOptionName, self.registry.register_lazy, '1,2', self.__module__, 'TestOptionRegistry.lazy_option')