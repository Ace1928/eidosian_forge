import json
import re
from unittest import mock
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import http_basic
from keystoneauth1 import noauth
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
class EndpointDataTests(utils.TestCase):

    def setUp(self):
        super(EndpointDataTests, self).setUp()
        self.session = session.Session()

    @mock.patch('keystoneauth1.discover.get_discovery')
    @mock.patch('keystoneauth1.discover.EndpointData._get_discovery_url_choices')
    def test_run_discovery_cache(self, mock_url_choices, mock_get_disc):
        mock_get_disc.side_effect = exceptions.DiscoveryFailure()
        mock_url_choices.return_value = ('url1', 'url2', 'url1', 'url3')
        epd = discover.EndpointData()
        epd._run_discovery(session='sess', cache='cache', min_version='min', max_version='max', project_id='projid', allow_version_hack='allow_hack', discover_versions='disc_vers')
        self.assertEqual(3, mock_get_disc.call_count)
        mock_get_disc.assert_has_calls([mock.call('sess', url, cache='cache', authenticated=False) for url in ('url1', 'url2', 'url3')])

    def test_run_discovery_auth(self):
        url = 'https://example.com'
        headers = {'Accept': 'application/json', 'OpenStack-API-Version': 'version header test'}
        session = mock.Mock()
        session.get.side_effect = [exceptions.Unauthorized('unauthorized'), exceptions.BadRequest('bad request')]
        try:
            discover.get_version_data(session, url, version_header='version header test')
        except exceptions.BadRequest:
            pass
        self.assertEqual(2, session.get.call_count)
        session.get.assert_has_calls([mock.call(url, headers=headers, authenticated=None), mock.call(url, headers=headers, authenticated=True)])

    def test_endpoint_data_str(self):
        """Validate EndpointData.__str__."""
        epd = discover.EndpointData(catalog_url='abc', service_type='123', api_version=(2, 3))
        exp = 'EndpointData{api_version=(2, 3), catalog_url=abc, endpoint_id=None, interface=None, major_version=None, max_microversion=None, min_microversion=None, next_min_version=None, not_before=None, raw_endpoint=None, region_name=None, service_id=None, service_name=None, service_type=123, service_url=None, url=abc}'
        self.assertEqual(exp, str(epd))
        self.assertEqual(exp, '%s' % epd)

    def test_project_id_int_fallback(self):
        bad_url = 'https://compute.example.com/v2/123456'
        epd = discover.EndpointData(catalog_url=bad_url)
        self.assertEqual((2, 0), epd.api_version)

    def test_url_version_match_project_id_int(self):
        self.session = session.Session()
        discovery_fixture = fixture.V3Discovery(V3_URL)
        discovery_doc = _create_single_version(discovery_fixture)
        self.requests_mock.get(V3_URL, status_code=200, json=discovery_doc)
        epd = discover.EndpointData(catalog_url=V3_URL).get_versioned_data(session=self.session, project_id='3')
        self.assertEqual(epd.catalog_url, epd.url)