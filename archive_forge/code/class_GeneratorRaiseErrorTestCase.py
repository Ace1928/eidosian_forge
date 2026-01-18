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
class GeneratorRaiseErrorTestCase(base.BaseTestCase):

    def test_generator_raises_error(self):
        """Verifies that errors from extension manager are not suppressed."""

        class FakeException(Exception):
            pass

        class FakeEP:

            def __init__(self):
                self.name = 'callback_is_expected'
                self.require = self.resolve
                self.load = self.resolve

            def resolve(self, *args, **kwargs):
                raise FakeException()
        fake_ep = FakeEP()
        self.conf = cfg.ConfigOpts()
        self.conf.register_opts(generator._generator_opts)
        self.conf.set_default('namespace', [fake_ep.name])
        with mock.patch('stevedore.named.NamedExtensionManager', side_effect=FakeException()):
            self.assertRaises(FakeException, generator.generate, self.conf)

    def test_generator_call_with_no_arguments_raises_system_exit(self):
        testargs = ['oslo-config-generator']
        with mock.patch('sys.argv', testargs):
            self.assertRaises(SystemExit, generator.main, [])