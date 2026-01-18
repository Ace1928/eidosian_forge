from unittest import mock
from blazarclient import base
from blazarclient import exception
from blazarclient import tests
class SessionClientTestCase(tests.TestCase):

    def setUp(self):
        super(SessionClientTestCase, self).setUp()
        self.manager = base.SessionClient(user_agent='python-blazarclient', session=mock.MagicMock())

    @mock.patch('blazarclient.base.adapter.LegacyJsonAdapter.request')
    def test_request_ok(self, m):
        mock_resp = mock.Mock()
        mock_resp.status_code = 200
        mock_body = {'resp_key': 'resp_value'}
        m.return_value = (mock_resp, mock_body)
        url = '/leases'
        kwargs = {'body': {'req_key': 'req_value'}}
        resp, body = self.manager.request(url, 'POST', **kwargs)
        self.assertEqual((resp, body), (mock_resp, mock_body))

    @mock.patch('blazarclient.base.adapter.LegacyJsonAdapter.request')
    def test_request_fail(self, m):
        resp = mock.Mock()
        resp.status_code = 400
        body = {'error message': 'error'}
        m.return_value = (resp, body)
        url = '/leases'
        kwargs = {'body': {'req_key': 'req_value'}}
        self.assertRaises(exception.BlazarClientException, self.manager.request, url, 'POST', **kwargs)