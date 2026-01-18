import requests
from heat_integrationtests.functional import functional_base
class VersionNegotiationTestCase(functional_base.FunctionalTestsBase):

    def test_authless_version_negotiation(self):
        heat_url = self.identity_client.get_endpoint_url('orchestration', region=self.conf.region)
        heat_api_root = heat_url.split('/v1')[0]
        expected_version_dict['versions'][0]['links'][0]['href'] = heat_api_root + '/v1/'
        r = requests.get(heat_api_root)
        self.assertEqual(300, r.status_code, 'got response %s' % r.text)
        self.assertEqual(expected_version_dict, r.json())