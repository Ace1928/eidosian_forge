import ddt
from keystoneauth1 import exceptions
from openstack.tests.unit import base
class TestBadPlacementRest(base.TestCase):

    def setUp(self):
        self.skipTest('Need to re-add support for broken placement versions')
        super(TestBadPlacementRest, self).setUp()
        self.use_placement(discovery_fixture='bad-placement.json')

    def _register_uris(self, status_code=None):
        uri = dict(method='GET', uri=self.get_mock_url('placement', 'public', append=['allocation_candidates']), json={})
        if status_code is not None:
            uri['status_code'] = status_code
        self.register_uris([uri])

    def _validate_resp(self, resp, status_code):
        self.assertEqual(status_code, resp.status_code)
        self.assertEqual('https://placement.example.com/allocation_candidates', resp.url)
        self.assert_calls()

    def test_discovery(self):
        self._register_uris()
        rs = self.cloud.placement.get('/allocation_candidates')
        self._validate_resp(rs, 200)