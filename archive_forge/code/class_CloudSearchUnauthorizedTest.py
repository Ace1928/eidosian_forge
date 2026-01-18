from tests.compat import mock, unittest
from httpretty import HTTPretty
import json
import requests
from boto.cloudsearch.search import SearchConnection, SearchServiceException
from boto.compat import six, map
class CloudSearchUnauthorizedTest(CloudSearchSearchBaseTest):
    response = b'<html><body><h1>403 Forbidden</h1>foo bar baz</body></html>'
    response_status = 403
    content_type = 'text/html'

    def test_response(self):
        search = SearchConnection(endpoint=HOSTNAME)
        with self.assertRaisesRegexp(SearchServiceException, 'foo bar baz'):
            search.search(q='Test')