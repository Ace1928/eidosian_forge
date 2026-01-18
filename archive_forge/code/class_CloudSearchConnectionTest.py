from tests.compat import mock, unittest
from httpretty import HTTPretty
import json
import requests
from boto.cloudsearch.search import SearchConnection, SearchServiceException
from boto.compat import six, map
class CloudSearchConnectionTest(unittest.TestCase):
    cloudsearch = True

    def setUp(self):
        super(CloudSearchConnectionTest, self).setUp()
        self.conn = SearchConnection(endpoint='test-domain.cloudsearch.amazonaws.com')

    def test_expose_additional_error_info(self):
        mpo = mock.patch.object
        fake = FakeResponse()
        fake.content = b'Nopenopenope'
        with mpo(requests, 'get', return_value=fake) as mock_request:
            with self.assertRaises(SearchServiceException) as cm:
                self.conn.search(q='not_gonna_happen')
            self.assertTrue('non-json response' in str(cm.exception))
            self.assertTrue('Nopenopenope' in str(cm.exception))
        fake.content = json.dumps({'error': 'Something went wrong. Oops.'}).encode('utf-8')
        with mpo(requests, 'get', return_value=fake) as mock_request:
            with self.assertRaises(SearchServiceException) as cm:
                self.conn.search(q='no_luck_here')
            self.assertTrue('Unknown error' in str(cm.exception))
            self.assertTrue('went wrong. Oops' in str(cm.exception))