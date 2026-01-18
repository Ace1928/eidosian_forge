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
class TestRegistryOption(TestOptionConverter):

    def get_option(self, registry):
        return config.RegistryOption('foo', registry, help='A registry option.')

    def test_convert_invalid(self):
        registry = _mod_registry.Registry()
        opt = self.get_option(registry)
        self.assertConvertInvalid(opt, [1])
        self.assertConvertInvalid(opt, 'notregistered')

    def test_convert_valid(self):
        registry = _mod_registry.Registry()
        registry.register('someval', 1234)
        opt = self.get_option(registry)
        self.assertConverted(1234, opt, 'someval')
        self.assertConverted(1234, opt, 'someval')
        self.assertConverted(None, opt, None)

    def test_help(self):
        registry = _mod_registry.Registry()
        registry.register('someval', 1234, help='some option')
        registry.register('dunno', 1234, help='some other option')
        opt = self.get_option(registry)
        self.assertEqual('A registry option.\n\nThe following values are supported:\n dunno - some other option\n someval - some option\n', opt.help)

    def test_get_help_text(self):
        registry = _mod_registry.Registry()
        registry.register('someval', 1234, help='some option')
        registry.register('dunno', 1234, help='some other option')
        opt = self.get_option(registry)
        self.assertEqual('A registry option.\n\nThe following values are supported:\n dunno - some other option\n someval - some option\n', opt.get_help_text())