import os
from oslotest import base
from requests import HTTPError
import requests_mock
import testtools
from oslo_config import _list_opts
from oslo_config import cfg
from oslo_config import fixture
from oslo_config import sources
from oslo_config.sources import _uri
class TestProcessingSources(base.BaseTestCase):

    def setUp(self):
        super(TestProcessingSources, self).setUp()
        self.conf = cfg.ConfigOpts()
        self.conf_fixture = self.useFixture(fixture.Config(self.conf))

    def test_no_sources_default(self):
        with base.mock.patch.object(self.conf, '_open_source_from_opt_group') as open_source:
            open_source.side_effect = AssertionError('should not be called')
            self.conf([])

    def test_no_sources(self):
        self.conf_fixture.config(config_source=[])
        with base.mock.patch.object(self.conf, '_open_source_from_opt_group') as open_source:
            open_source.side_effect = AssertionError('should not be called')
            self.conf([])

    def test_source_named(self):
        self.conf_fixture.config(config_source=['missing_source'])
        with base.mock.patch.object(self.conf, '_open_source_from_opt_group') as open_source:
            self.conf([])
            open_source.assert_called_once_with('missing_source')

    def test_multiple_sources_named(self):
        self.conf_fixture.config(config_source=['source1', 'source2'])
        with base.mock.patch.object(self.conf, '_open_source_from_opt_group') as open_source:
            self.conf([])
            open_source.assert_has_calls([base.mock.call('source1'), base.mock.call('source2')])