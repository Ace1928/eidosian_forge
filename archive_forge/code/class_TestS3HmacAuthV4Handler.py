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
class TestS3HmacAuthV4Handler(unittest.TestCase):

    def setUp(self):
        self.provider = mock.Mock()
        self.provider.access_key = 'access_key'
        self.provider.secret_key = 'secret_key'
        self.provider.security_token = 'sekret_tokens'
        self.request = HTTPRequest('GET', 'https', 's3-us-west-2.amazonaws.com', 443, '/awesome-bucket/?max-keys=0', None, {}, {}, '')
        self.awesome_bucket_request = HTTPRequest(method='GET', protocol='https', host='awesome-bucket.s3-us-west-2.amazonaws.com', port=443, path='/', auth_path=None, params={'max-keys': 0}, headers={'User-Agent': 'Boto', 'X-AMZ-Content-sha256': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'X-AMZ-Date': '20130605T193245Z'}, body='')
        self.auth = S3HmacAuthV4Handler(host='awesome-bucket.s3-us-west-2.amazonaws.com', config=mock.Mock(), provider=self.provider, region_name='s3-us-west-2')

    def test_clean_region_name(self):
        cleaned = self.auth.clean_region_name('us-west-2')
        self.assertEqual(cleaned, 'us-west-2')
        cleaned = self.auth.clean_region_name('s3-us-west-2')
        self.assertEqual(cleaned, 'us-west-2')
        cleaned = self.auth.clean_region_name('s3.amazonaws.com')
        self.assertEqual(cleaned, 's3.amazonaws.com')
        cleaned = self.auth.clean_region_name('something-s3-us-west-2')
        self.assertEqual(cleaned, 'something-s3-us-west-2')

    def test_region_stripping(self):
        auth = S3HmacAuthV4Handler(host='s3-us-west-2.amazonaws.com', config=mock.Mock(), provider=self.provider)
        self.assertEqual(auth.region_name, None)
        auth = S3HmacAuthV4Handler(host='s3-us-west-2.amazonaws.com', config=mock.Mock(), provider=self.provider, region_name='us-west-2')
        self.assertEqual(auth.region_name, 'us-west-2')
        self.assertEqual(self.auth.region_name, 'us-west-2')

    def test_determine_region_name(self):
        name = self.auth.determine_region_name('s3-us-west-2.amazonaws.com')
        self.assertEqual(name, 'us-west-2')

    def test_canonical_uri(self):
        request = HTTPRequest('GET', 'https', 's3-us-west-2.amazonaws.com', 443, 'x/./././~x .html', None, {}, {}, '')
        canonical_uri = self.auth.canonical_uri(request)
        self.assertEqual(canonical_uri, 'x/./././~x%20.html')

    def test_determine_service_name(self):
        name = self.auth.determine_service_name('s3.us-west-2.amazonaws.com')
        self.assertEqual(name, 's3')
        name = self.auth.determine_service_name('s3-us-west-2.amazonaws.com')
        self.assertEqual(name, 's3')
        name = self.auth.determine_service_name('bucket.s3.us-west-2.amazonaws.com')
        self.assertEqual(name, 's3')
        name = self.auth.determine_service_name('bucket.s3-us-west-2.amazonaws.com')
        self.assertEqual(name, 's3')

    def test_add_auth(self):
        self.assertFalse('x-amz-content-sha256' in self.request.headers)
        self.auth.add_auth(self.request)
        self.assertTrue('x-amz-content-sha256' in self.request.headers)
        the_sha = self.request.headers['x-amz-content-sha256']
        self.assertEqual(the_sha, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855')

    def test_host_header(self):
        host = self.auth.host_header(self.awesome_bucket_request.host, self.awesome_bucket_request)
        self.assertEqual(host, 'awesome-bucket.s3-us-west-2.amazonaws.com')

    def test_canonical_query_string(self):
        qs = self.auth.canonical_query_string(self.awesome_bucket_request)
        self.assertEqual(qs, 'max-keys=0')

    def test_correct_handling_of_plus_sign(self):
        request = HTTPRequest('GET', 'https', 's3-us-west-2.amazonaws.com', 443, 'hello+world.txt', None, {}, {}, '')
        canonical_uri = self.auth.canonical_uri(request)
        self.assertEqual(canonical_uri, 'hello%2Bworld.txt')
        request = HTTPRequest('GET', 'https', 's3-us-west-2.amazonaws.com', 443, 'hello%2Bworld.txt', None, {}, {}, '')
        canonical_uri = self.auth.canonical_uri(request)
        self.assertEqual(canonical_uri, 'hello%2Bworld.txt')

    def test_mangle_path_and_params(self):
        request = HTTPRequest(method='GET', protocol='https', host='awesome-bucket.s3-us-west-2.amazonaws.com', port=443, path='/?delete&max-keys=0', auth_path=None, params={'key': 'why hello there', 'max-keys': 1}, headers={'User-Agent': 'Boto', 'X-AMZ-Content-sha256': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'X-AMZ-Date': '20130605T193245Z'}, body='')
        mod_req = self.auth.mangle_path_and_params(request)
        self.assertEqual(mod_req.path, '/?delete&max-keys=0')
        self.assertEqual(mod_req.auth_path, '/')
        self.assertEqual(mod_req.params, {'max-keys': '0', 'key': 'why hello there', 'delete': ''})

    def test_unicode_query_string(self):
        request = HTTPRequest(method='HEAD', protocol='https', host='awesome-bucket.s3-us-west-2.amazonaws.com', port=443, path=u'/?max-keys=1&prefix=El%20Ni%C3%B1o', auth_path=u'/awesome-bucket/?max-keys=1&prefix=El%20Ni%C3%B1o', params={}, headers={}, body='')
        mod_req = self.auth.mangle_path_and_params(request)
        self.assertEqual(mod_req.path, u'/?max-keys=1&prefix=El%20Ni%C3%B1o')
        self.assertEqual(mod_req.auth_path, u'/awesome-bucket/')
        self.assertEqual(mod_req.params, {u'max-keys': u'1', u'prefix': u'El Niño'})

    def test_canonical_request(self):
        expected = 'GET\n/\nmax-keys=0\nhost:awesome-bucket.s3-us-west-2.amazonaws.com\nuser-agent:Boto\nx-amz-content-sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\nx-amz-date:20130605T193245Z\n\nhost;user-agent;x-amz-content-sha256;x-amz-date\ne3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
        authed_req = self.auth.canonical_request(self.awesome_bucket_request)
        self.assertEqual(authed_req, expected)
        request = copy.copy(self.awesome_bucket_request)
        request.path = request.auth_path = '/?max-keys=0'
        request.params = {}
        expected = 'GET\n/\nmax-keys=0\nhost:awesome-bucket.s3-us-west-2.amazonaws.com\nuser-agent:Boto\nx-amz-content-sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\nx-amz-date:20130605T193245Z\n\nhost;user-agent;x-amz-content-sha256;x-amz-date\ne3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
        request = self.auth.mangle_path_and_params(request)
        authed_req = self.auth.canonical_request(request)
        self.assertEqual(authed_req, expected)

    def test_non_string_headers(self):
        self.awesome_bucket_request.headers['Content-Length'] = 8
        self.awesome_bucket_request.headers['x-amz-server-side-encryption-customer-key-md5'] = 2
        self.awesome_bucket_request.headers['x-amz-server-side-encryption-customer-key'] = 1
        canonical_headers = self.auth.canonical_headers(self.awesome_bucket_request.headers)
        self.assertEqual(canonical_headers, 'content-length:8\nuser-agent:Boto\nx-amz-content-sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\nx-amz-date:20130605T193245Z\nx-amz-server-side-encryption-customer-key:1\nx-amz-server-side-encryption-customer-key-md5:2')