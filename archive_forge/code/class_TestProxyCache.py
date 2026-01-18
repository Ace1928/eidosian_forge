import copy
import queue
from unittest import mock
from keystoneauth1 import session
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack import proxy
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
class TestProxyCache(base.TestCase):

    class Res(resource.Resource):
        base_path = 'fake'
        allow_commit = True
        allow_fetch = True
        foo = resource.Body('foo')

    def setUp(self):
        super(TestProxyCache, self).setUp(cloud_config_fixture='clouds_cache.yaml')
        self.session = mock.Mock(spec=session.Session)
        self.session._sdk_connection = self.cloud
        self.session.get_project_id = mock.Mock(return_value='fake_prj')
        self.response = mock.Mock()
        self.response.status_code = 200
        self.response.history = []
        self.response.headers = {}
        self.response.body = {}
        self.response.json = mock.Mock(return_value=self.response.body)
        self.session.request = mock.Mock(return_value=self.response)
        self.sot = proxy.Proxy(self.session)
        self.sot._connection = self.cloud
        self.sot.service_type = 'srv'

    def _get_key(self, id):
        return "srv.fake.fake/%s.{'microversion': None, 'params': {}}" % id

    def test_get_not_in_cache(self):
        self.cloud._cache_expirations['srv.fake'] = 5
        self.sot._get(self.Res, '1')
        self.session.request.assert_called_with('fake/1', 'GET', connect_retries=mock.ANY, raise_exc=mock.ANY, global_request_id=mock.ANY, microversion=mock.ANY, params=mock.ANY, endpoint_filter=mock.ANY, headers=mock.ANY, rate_semaphore=mock.ANY)
        self.assertIn(self._get_key(1), self.cloud._api_cache_keys)

    def test_get_from_cache(self):
        key = self._get_key(2)
        self.cloud._cache.set(key, self.response)
        self.cloud._cache_expirations['srv.fake'] = 5
        self.sot._get(self.Res, '2')
        self.session.request.assert_not_called()

    def test_modify(self):
        key = self._get_key(3)
        self.cloud._cache.set(key, self.response)
        self.cloud._api_cache_keys.add(key)
        self.cloud._cache_expirations['srv.fake'] = 5
        self.sot._get(self.Res, '3')
        self.session.request.assert_not_called()
        rs = self.Res.existing(id='3')
        self.sot._update(self.Res, rs, foo='bar')
        self.session.request.assert_called()
        self.assertIsNotNone(self.cloud._cache.get(key))
        self.assertEqual('NoValue', type(self.cloud._cache.get(key)).__name__)
        self.assertNotIn(key, self.cloud._api_cache_keys)
        self.sot._get(self.Res, '3')
        self.session.request.assert_called()

    def test_get_bypass_cache(self):
        key = self._get_key(4)
        resp = copy.deepcopy(self.response)
        resp.body = {'foo': 'bar'}
        self.cloud._api_cache_keys.add(key)
        self.cloud._cache.set(key, resp)
        self.cloud._cache_expirations['srv.fake'] = 5
        self.sot._get(self.Res, '4', skip_cache=True)
        self.session.request.assert_called()
        self.assertEqual(dict(), self.response.body)
        self.assertNotIn(key, self.cloud._api_cache_keys)
        self.assertEqual('NoValue', type(self.cloud._cache.get(key)).__name__)