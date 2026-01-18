from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.tests.api import fakes as api_fakes
class TestBaseAPICreate(api_fakes.TestSession):

    def setUp(self):
        super(TestBaseAPICreate, self).setUp()
        self.api = api.BaseAPI(session=self.sess, endpoint=self.BASE_URL)

    def test_baseapi_create_post(self):
        self.requests_mock.register_uri('POST', self.BASE_URL + '/qaz', json=api_fakes.RESP_ITEM_1, status_code=202)
        ret = self.api.create('qaz')
        self.assertEqual(api_fakes.RESP_ITEM_1, ret)

    def test_baseapi_create_put(self):
        self.requests_mock.register_uri('PUT', self.BASE_URL + '/qaz', json=api_fakes.RESP_ITEM_1, status_code=202)
        ret = self.api.create('qaz', method='PUT')
        self.assertEqual(api_fakes.RESP_ITEM_1, ret)

    def test_baseapi_delete(self):
        self.requests_mock.register_uri('DELETE', self.BASE_URL + '/qaz', status_code=204)
        ret = self.api.delete('qaz')
        self.assertEqual(204, ret.status_code)