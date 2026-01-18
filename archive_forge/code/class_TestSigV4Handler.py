import copy
import pickle
import os
from tests.compat import unittest, mock
from tests.unit import MockServiceWithConfigTestCase
from nose.tools import assert_equal
from boto.auth import HmacAuthV4Handler
from boto.auth import S3HmacAuthV4Handler
from boto.auth import detect_potential_s3sigv4
from boto.auth import detect_potential_sigv4
from boto.connection import HTTPRequest
from boto.provider import Provider
from boto.regioninfo import RegionInfo
class TestSigV4Handler(unittest.TestCase):

    def setUp(self):
        self.provider = mock.Mock()
        self.provider.access_key = 'access_key'
        self.provider.secret_key = 'secret_key'
        self.request = HTTPRequest('POST', 'https', 'glacier.us-east-1.amazonaws.com', 443, '/-/vaults/foo/archives', None, {}, {'x-amz-glacier-version': '2012-06-01'}, '')

    def test_not_adding_empty_qs(self):
        self.provider.security_token = None
        auth = HmacAuthV4Handler('glacier.us-east-1.amazonaws.com', mock.Mock(), self.provider)
        req = copy.copy(self.request)
        auth.add_auth(req)
        self.assertEqual(req.path, '/-/vaults/foo/archives')

    def test_inner_whitespace_is_collapsed(self):
        auth = HmacAuthV4Handler('glacier.us-east-1.amazonaws.com', mock.Mock(), self.provider)
        self.request.headers['x-amz-archive-description'] = 'two  spaces'
        self.request.headers['x-amz-quoted-string'] = '  "a   b   c" '
        headers = auth.headers_to_sign(self.request)
        self.assertEqual(headers, {'Host': 'glacier.us-east-1.amazonaws.com', 'x-amz-archive-description': 'two  spaces', 'x-amz-glacier-version': '2012-06-01', 'x-amz-quoted-string': '  "a   b   c" '})
        self.assertEqual(auth.canonical_headers(headers), 'host:glacier.us-east-1.amazonaws.com\nx-amz-archive-description:two spaces\nx-amz-glacier-version:2012-06-01\nx-amz-quoted-string:"a   b   c"')

    def test_canonical_query_string(self):
        auth = HmacAuthV4Handler('glacier.us-east-1.amazonaws.com', mock.Mock(), self.provider)
        request = HTTPRequest('GET', 'https', 'glacier.us-east-1.amazonaws.com', 443, '/-/vaults/foo/archives', None, {}, {'x-amz-glacier-version': '2012-06-01'}, '')
        request.params['Foo.1'] = 'aaa'
        request.params['Foo.10'] = 'zzz'
        query_string = auth.canonical_query_string(request)
        self.assertEqual(query_string, 'Foo.1=aaa&Foo.10=zzz')

    def test_query_string(self):
        auth = HmacAuthV4Handler('sns.us-east-1.amazonaws.com', mock.Mock(), self.provider)
        params = {'Message': u'We ♥ utf-8'.encode('utf-8')}
        request = HTTPRequest('POST', 'https', 'sns.us-east-1.amazonaws.com', 443, '/', None, params, {}, '')
        query_string = auth.query_string(request)
        self.assertEqual(query_string, 'Message=We%20%E2%99%A5%20utf-8')

    def test_canonical_uri(self):
        auth = HmacAuthV4Handler('glacier.us-east-1.amazonaws.com', mock.Mock(), self.provider)
        request = HTTPRequest('GET', 'https', 'glacier.us-east-1.amazonaws.com', 443, 'x/./././x .html', None, {}, {'x-amz-glacier-version': '2012-06-01'}, '')
        canonical_uri = auth.canonical_uri(request)
        self.assertEqual(canonical_uri, 'x/x%20.html')
        auth = HmacAuthV4Handler('glacier.us-east-1.amazonaws.com', mock.Mock(), self.provider)
        request = HTTPRequest('GET', 'https', 'glacier.us-east-1.amazonaws.com', 443, 'x/./././x/html/', None, {}, {'x-amz-glacier-version': '2012-06-01'}, '')
        canonical_uri = auth.canonical_uri(request)
        self.assertEqual(canonical_uri, 'x/x/html/')
        request = HTTPRequest('GET', 'https', 'glacier.us-east-1.amazonaws.com', 443, '/', None, {}, {'x-amz-glacier-version': '2012-06-01'}, '')
        canonical_uri = auth.canonical_uri(request)
        self.assertEqual(canonical_uri, '/')
        request = HTTPRequest('GET', 'https', 'glacier.us-east-1.amazonaws.com', 443, '\\x\\x.html', None, {}, {'x-amz-glacier-version': '2012-06-01'}, '')
        canonical_uri = auth.canonical_uri(request)
        self.assertEqual(canonical_uri, '/x/x.html')

    def test_credential_scope(self):
        auth = HmacAuthV4Handler('iam.amazonaws.com', mock.Mock(), self.provider)
        request = HTTPRequest('POST', 'https', 'iam.amazonaws.com', 443, '/', '/', {'Action': 'ListAccountAliases', 'Version': '2010-05-08'}, {'Content-Length': '44', 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8', 'X-Amz-Date': '20130808T013210Z'}, 'Action=ListAccountAliases&Version=2010-05-08')
        credential_scope = auth.credential_scope(request)
        region_name = credential_scope.split('/')[1]
        self.assertEqual(region_name, 'us-east-1')
        auth = HmacAuthV4Handler('iam.us-gov.amazonaws.com', mock.Mock(), self.provider)
        request = HTTPRequest('POST', 'https', 'iam.us-gov.amazonaws.com', 443, '/', '/', {'Action': 'ListAccountAliases', 'Version': '2010-05-08'}, {'Content-Length': '44', 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8', 'X-Amz-Date': '20130808T013210Z'}, 'Action=ListAccountAliases&Version=2010-05-08')
        credential_scope = auth.credential_scope(request)
        region_name = credential_scope.split('/')[1]
        self.assertEqual(region_name, 'us-gov-west-1')
        auth = HmacAuthV4Handler('iam.us-west-1.amazonaws.com', mock.Mock(), self.provider)
        request = HTTPRequest('POST', 'https', 'iam.us-west-1.amazonaws.com', 443, '/', '/', {'Action': 'ListAccountAliases', 'Version': '2010-05-08'}, {'Content-Length': '44', 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8', 'X-Amz-Date': '20130808T013210Z'}, 'Action=ListAccountAliases&Version=2010-05-08')
        credential_scope = auth.credential_scope(request)
        region_name = credential_scope.split('/')[1]
        self.assertEqual(region_name, 'us-west-1')
        auth = HmacAuthV4Handler('localhost', mock.Mock(), self.provider, service_name='iam')
        request = HTTPRequest('POST', 'http', 'localhost', 8080, '/', '/', {'Action': 'ListAccountAliases', 'Version': '2010-05-08'}, {'Content-Length': '44', 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8', 'X-Amz-Date': '20130808T013210Z'}, 'Action=ListAccountAliases&Version=2010-05-08')
        credential_scope = auth.credential_scope(request)
        timestamp, region, service, v = credential_scope.split('/')
        self.assertEqual(region, 'localhost')
        self.assertEqual(service, 'iam')

    def test_headers_to_sign(self):
        auth = HmacAuthV4Handler('glacier.us-east-1.amazonaws.com', mock.Mock(), self.provider)
        request = HTTPRequest('GET', 'http', 'glacier.us-east-1.amazonaws.com', 80, 'x/./././x .html', None, {}, {'x-amz-glacier-version': '2012-06-01'}, '')
        headers = auth.headers_to_sign(request)
        self.assertEqual(headers['Host'], 'glacier.us-east-1.amazonaws.com')
        request = HTTPRequest('GET', 'https', 'glacier.us-east-1.amazonaws.com', 443, 'x/./././x .html', None, {}, {'x-amz-glacier-version': '2012-06-01'}, '')
        headers = auth.headers_to_sign(request)
        self.assertEqual(headers['Host'], 'glacier.us-east-1.amazonaws.com')
        request = HTTPRequest('GET', 'https', 'glacier.us-east-1.amazonaws.com', 8080, 'x/./././x .html', None, {}, {'x-amz-glacier-version': '2012-06-01'}, '')
        headers = auth.headers_to_sign(request)
        self.assertEqual(headers['Host'], 'glacier.us-east-1.amazonaws.com:8080')

    def test_region_and_service_can_be_overriden(self):
        auth = HmacAuthV4Handler('queue.amazonaws.com', mock.Mock(), self.provider)
        self.request.headers['X-Amz-Date'] = '20121121000000'
        auth.region_name = 'us-west-2'
        auth.service_name = 'sqs'
        scope = auth.credential_scope(self.request)
        self.assertEqual(scope, '20121121/us-west-2/sqs/aws4_request')

    def test_pickle_works(self):
        provider = Provider('aws', access_key='access_key', secret_key='secret_key')
        auth = HmacAuthV4Handler('queue.amazonaws.com', None, provider)
        pickled = pickle.dumps(auth)
        auth2 = pickle.loads(pickled)
        self.assertEqual(auth.host, auth2.host)

    def test_bytes_header(self):
        auth = HmacAuthV4Handler('glacier.us-east-1.amazonaws.com', mock.Mock(), self.provider)
        request = HTTPRequest('GET', 'http', 'glacier.us-east-1.amazonaws.com', 80, 'x/./././x .html', None, {}, {'x-amz-glacier-version': '2012-06-01', 'x-amz-hash': b'f00'}, '')
        canonical = auth.canonical_request(request)
        self.assertIn('f00', canonical)