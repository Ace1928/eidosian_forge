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
class ChangeDefaultsTestCase(base.BaseTestCase):

    @mock.patch.object(generator, '_get_opt_default_updaters')
    @mock.patch.object(generator, '_get_raw_opts_loaders')
    def test_no_modifiers_registered(self, raw_opts_loaders, get_updaters):
        orig_opt = cfg.StrOpt('foo', default='bar')
        raw_opts_loaders.return_value = [('namespace', lambda: [(None, [orig_opt])])]
        get_updaters.return_value = []
        opts = generator._list_opts(['namespace'])
        the_opt = opts[0][1][0][1][0]
        self.assertEqual('bar', the_opt.default)
        self.assertIs(orig_opt, the_opt)

    @mock.patch.object(generator, '_get_opt_default_updaters')
    @mock.patch.object(generator, '_get_raw_opts_loaders')
    def test_change_default(self, raw_opts_loaders, get_updaters):
        orig_opt = cfg.StrOpt('foo', default='bar')
        raw_opts_loaders.return_value = [('namespace', lambda: [(None, [orig_opt])])]

        def updater():
            cfg.set_defaults([orig_opt], foo='blah')
        get_updaters.return_value = [updater]
        opts = generator._list_opts(['namespace'])
        the_opt = opts[0][1][0][1][0]
        self.assertEqual('blah', the_opt.default)
        self.assertIs(orig_opt, the_opt)