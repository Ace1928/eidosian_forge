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
class TestLoading(base.BaseTestCase):

    def setUp(self):
        super(TestLoading, self).setUp()
        self.conf = cfg.ConfigOpts()
        self.conf_fixture = self.useFixture(fixture.Config(self.conf))

    def test_source_missing(self):
        source = self.conf._open_source_from_opt_group('missing_source')
        self.assertIsNone(source)

    def test_driver_missing(self):
        self.conf_fixture.load_raw_values(group='missing_driver', not_driver='foo')
        source = self.conf._open_source_from_opt_group('missing_driver')
        self.assertIsNone(source)

    def test_unknown_driver(self):
        self.conf_fixture.load_raw_values(group='unknown_driver', driver='very_unlikely_to_exist_driver_name')
        source = self.conf._open_source_from_opt_group('unknown_driver')
        self.assertIsNone(source)