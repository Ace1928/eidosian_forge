from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.tests.api import fakes as api_fakes
class TestBaseAPIFind(api_fakes.TestSession):

    def setUp(self):
        super(TestBaseAPIFind, self).setUp()
        self.api = api.BaseAPI(session=self.sess, endpoint=self.BASE_URL)

    def test_baseapi_find(self):
        self.requests_mock.register_uri('GET', self.BASE_URL + '/qaz/1', json={'qaz': api_fakes.RESP_ITEM_1}, status_code=200)
        ret = self.api.find('qaz', '1')
        self.assertEqual(api_fakes.RESP_ITEM_1, ret)
        self.requests_mock.register_uri('GET', self.BASE_URL + '/qaz/1', status_code=404)
        self.assertRaises(exceptions.NotFound, self.api.find, 'qaz', '1')

    def test_baseapi_find_attr_by_id(self):
        self.requests_mock.register_uri('GET', self.BASE_URL + '/qaz?name=1', json={'qaz': []}, status_code=200)
        self.requests_mock.register_uri('GET', self.BASE_URL + '/qaz?id=1', json={'qaz': [api_fakes.RESP_ITEM_1]}, status_code=200)
        ret = self.api.find_attr('qaz', '1')
        self.assertEqual(api_fakes.RESP_ITEM_1, ret)
        self.requests_mock.register_uri('GET', self.BASE_URL + '/qaz?name=0', json={'qaz': []}, status_code=200)
        self.requests_mock.register_uri('GET', self.BASE_URL + '/qaz?id=0', json={'qaz': []}, status_code=200)
        self.assertRaises(exceptions.CommandError, self.api.find_attr, 'qaz', '0')
        self.requests_mock.register_uri('GET', self.BASE_URL + '/qaz?status=UP', json={'qaz': [api_fakes.RESP_ITEM_1]}, status_code=200)
        ret = self.api.find_attr('qaz', 'UP', attr='status')
        self.assertEqual(api_fakes.RESP_ITEM_1, ret)
        ret = self.api.find_attr('qaz', value='UP', attr='status')
        self.assertEqual(api_fakes.RESP_ITEM_1, ret)

    def test_baseapi_find_attr_by_name(self):
        self.requests_mock.register_uri('GET', self.BASE_URL + '/qaz?name=alpha', json={'qaz': [api_fakes.RESP_ITEM_1]}, status_code=200)
        ret = self.api.find_attr('qaz', 'alpha')
        self.assertEqual(api_fakes.RESP_ITEM_1, ret)
        self.requests_mock.register_uri('GET', self.BASE_URL + '/qaz?name=0', json={'qaz': []}, status_code=200)
        self.requests_mock.register_uri('GET', self.BASE_URL + '/qaz?id=0', json={'qaz': []}, status_code=200)
        self.assertRaises(exceptions.CommandError, self.api.find_attr, 'qaz', '0')
        self.requests_mock.register_uri('GET', self.BASE_URL + '/qaz?status=UP', json={'qaz': [api_fakes.RESP_ITEM_1]}, status_code=200)
        ret = self.api.find_attr('qaz', 'UP', attr='status')
        self.assertEqual(api_fakes.RESP_ITEM_1, ret)
        ret = self.api.find_attr('qaz', value='UP', attr='status')
        self.assertEqual(api_fakes.RESP_ITEM_1, ret)

    def test_baseapi_find_attr_path_resource(self):
        self.requests_mock.register_uri('GET', self.BASE_URL + '/wsx?name=1', json={'qaz': []}, status_code=200)
        self.requests_mock.register_uri('GET', self.BASE_URL + '/wsx?id=1', json={'qaz': [api_fakes.RESP_ITEM_1]}, status_code=200)
        ret = self.api.find_attr('wsx', '1', resource='qaz')
        self.assertEqual(api_fakes.RESP_ITEM_1, ret)

    def test_baseapi_find_bulk_none(self):
        self.requests_mock.register_uri('GET', self.BASE_URL + '/qaz', json=api_fakes.LIST_RESP, status_code=200)
        ret = self.api.find_bulk('qaz')
        self.assertEqual(api_fakes.LIST_RESP, ret)
        ret = self.api.find_bulk('qaz', headers={})
        self.assertEqual(api_fakes.LIST_RESP, ret)

    def test_baseapi_find_bulk_one(self):
        self.requests_mock.register_uri('GET', self.BASE_URL + '/qaz', json=api_fakes.LIST_RESP, status_code=200)
        ret = self.api.find_bulk('qaz', id='1')
        self.assertEqual([api_fakes.LIST_RESP[0]], ret)
        ret = self.api.find_bulk('qaz', id='1', headers={})
        self.assertEqual([api_fakes.LIST_RESP[0]], ret)
        ret = self.api.find_bulk('qaz', id='0')
        self.assertEqual([], ret)
        ret = self.api.find_bulk('qaz', name='beta')
        self.assertEqual([api_fakes.LIST_RESP[1]], ret)
        ret = self.api.find_bulk('qaz', error='bogus')
        self.assertEqual([], ret)

    def test_baseapi_find_bulk_two(self):
        self.requests_mock.register_uri('GET', self.BASE_URL + '/qaz', json=api_fakes.LIST_RESP, status_code=200)
        ret = self.api.find_bulk('qaz', id='1', name='alpha')
        self.assertEqual([api_fakes.LIST_RESP[0]], ret)
        ret = self.api.find_bulk('qaz', id='1', name='beta')
        self.assertEqual([], ret)
        ret = self.api.find_bulk('qaz', id='1', error='beta')
        self.assertEqual([], ret)

    def test_baseapi_find_bulk_dict(self):
        self.requests_mock.register_uri('GET', self.BASE_URL + '/qaz', json={'qaz': api_fakes.LIST_RESP}, status_code=200)
        ret = self.api.find_bulk('qaz', id='1')
        self.assertEqual([api_fakes.LIST_RESP[0]], ret)