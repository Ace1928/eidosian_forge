import ddt
from keystoneauth1 import exceptions
from openstack.tests.unit import base
@ddt.ddt
class TestPlacementRest(base.TestCase):

    def setUp(self):
        super(TestPlacementRest, self).setUp()
        self.use_placement()

    def _register_uris(self, status_code=None):
        uri = dict(method='GET', uri=self.get_mock_url('placement', 'public', append=['allocation_candidates']), json={})
        if status_code is not None:
            uri['status_code'] = status_code
        self.register_uris([uri])

    def _validate_resp(self, resp, status_code):
        self.assertEqual(status_code, resp.status_code)
        self.assertEqual('https://placement.example.com/allocation_candidates', resp.url)
        self.assert_calls()

    @ddt.data({}, {'raise_exc': False}, {'raise_exc': True})
    def test_discovery(self, get_kwargs):
        self._register_uris()
        rs = self.cloud.placement.get('/allocation_candidates', **get_kwargs)
        self._validate_resp(rs, 200)

    @ddt.data({}, {'raise_exc': False})
    def test_discovery_err(self, get_kwargs):
        self._register_uris(status_code=500)
        rs = self.cloud.placement.get('/allocation_candidates', **get_kwargs)
        self._validate_resp(rs, 500)

    def test_discovery_exc(self):
        self._register_uris(status_code=500)
        ex = self.assertRaises(exceptions.InternalServerError, self.cloud.placement.get, '/allocation_candidates', raise_exc=True)
        self._validate_resp(ex.response, 500)

    def test_microversion_discovery(self):
        self.assertEqual((1, 17), self.cloud.placement.get_endpoint_data().max_microversion)
        self.assert_calls()