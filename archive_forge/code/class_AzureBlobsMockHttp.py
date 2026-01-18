import os
import sys
import json
import tempfile
from io import BytesIO
from libcloud.test import generate_random_data  # pylint: disable-msg=E0611
from libcloud.test import unittest
from libcloud.utils.py3 import b, httplib, parse_qs, urlparse, basestring
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_AZURE_BLOBS_PARAMS, STORAGE_AZURITE_BLOBS_PARAMS
from libcloud.storage.types import (
from libcloud.test.storage.base import BaseRangeDownloadMockHttp
from libcloud.test.file_fixtures import StorageFileFixtures  # pylint: disable-msg=E0611
from libcloud.storage.drivers.azure_blobs import (
class AzureBlobsMockHttp(BaseRangeDownloadMockHttp, unittest.TestCase):
    fixtures = StorageFileFixtures('azure_blobs')
    base_headers = {}

    def __getattr__(self, n):

        def fn(method, url, body, headers):
            fixture = self.fixtures.load(n + '.json')
            if method in ('POST', 'PUT'):
                try:
                    body = json.loads(body)
                    fixture_tmp = json.loads(fixture)
                    fixture_tmp = self._update(fixture_tmp, body)
                    fixture = json.dumps(fixture_tmp)
                except ValueError:
                    pass
            return (httplib.OK, fixture, headers, httplib.responses[httplib.OK])
        return fn

    def _UNAUTHORIZED(self, method, url, body, headers):
        return (httplib.UNAUTHORIZED, '', self.base_headers, httplib.responses[httplib.UNAUTHORIZED])

    def _list_containers_EMPTY(self, method, url, body, headers):
        body = self.fixtures.load('list_containers_empty.xml')
        return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])

    def _list_containers(self, method, url, body, headers):
        query_string = urlparse.urlsplit(url).query
        query = parse_qs(query_string)
        if 'marker' not in query:
            body = self.fixtures.load('list_containers_1.xml')
        else:
            body = self.fixtures.load('list_containers_2.xml')
        return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])

    def _test_container_EMPTY(self, method, url, body, headers):
        if method == 'DELETE':
            body = ''
            return (httplib.ACCEPTED, body, self.base_headers, httplib.responses[httplib.ACCEPTED])
        else:
            body = self.fixtures.load('list_objects_empty.xml')
            return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])

    def _new__container_INVALID_NAME(self, method, url, body, headers):
        return (httplib.BAD_REQUEST, body, self.base_headers, httplib.responses[httplib.BAD_REQUEST])

    def _test_container(self, method, url, body, headers):
        query_string = urlparse.urlsplit(url).query
        query = parse_qs(query_string)
        if 'marker' not in query:
            body = self.fixtures.load('list_objects_1.xml')
        else:
            body = self.fixtures.load('list_objects_2.xml')
        return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])

    def _test_container100(self, method, url, body, headers):
        body = ''
        if method != 'HEAD':
            return (httplib.BAD_REQUEST, body, self.base_headers, httplib.responses[httplib.BAD_REQUEST])
        return (httplib.NOT_FOUND, body, self.base_headers, httplib.responses[httplib.NOT_FOUND])

    def _test_container200(self, method, url, body, headers):
        body = ''
        if method != 'HEAD':
            return (httplib.BAD_REQUEST, body, self.base_headers, httplib.responses[httplib.BAD_REQUEST])
        headers = {}
        headers['etag'] = '0x8CFB877BB56A6FB'
        headers['last-modified'] = 'Fri, 04 Jan 2013 09:48:06 GMT'
        headers['x-ms-lease-status'] = 'unlocked'
        headers['x-ms-lease-state'] = 'available'
        headers['x-ms-meta-meta1'] = 'value1'
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _test_container200_test(self, method, url, body, headers):
        body = ''
        if method != 'HEAD':
            return (httplib.BAD_REQUEST, body, self.base_headers, httplib.responses[httplib.BAD_REQUEST])
        headers = {}
        headers['etag'] = '0x8CFB877BB56A6FB'
        headers['last-modified'] = 'Fri, 04 Jan 2013 09:48:06 GMT'
        headers['content-length'] = '12345'
        headers['content-type'] = 'application/zip'
        headers['x-ms-blob-type'] = 'Block'
        headers['x-ms-lease-status'] = 'unlocked'
        headers['x-ms-lease-state'] = 'available'
        headers['x-ms-meta-rabbits'] = 'monkeys'
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _test2_test_list_containers(self, method, url, body, headers):
        body = self.fixtures.load('list_containers.xml')
        headers = {'content-type': 'application/zip', 'etag': '"e31208wqsdoj329jd"', 'x-amz-meta-rabbits': 'monkeys', 'content-length': '12345', 'last-modified': 'Thu, 13 Sep 2012 07:13:22 GMT'}
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _new_container_ALREADY_EXISTS(self, method, url, body, headers):
        return (httplib.CONFLICT, body, headers, httplib.responses[httplib.CONFLICT])

    def _new_container(self, method, url, body, headers):
        headers = {}
        if method == 'PUT':
            status = httplib.CREATED
            headers['etag'] = '0x8CFB877BB56A6FB'
            headers['last-modified'] = 'Fri, 04 Jan 2013 09:48:06 GMT'
            headers['x-ms-lease-status'] = 'unlocked'
            headers['x-ms-lease-state'] = 'available'
            headers['x-ms-meta-meta1'] = 'value1'
        elif method == 'DELETE':
            status = httplib.NO_CONTENT
        return (status, body, headers, httplib.responses[status])

    def _new_container_DOESNT_EXIST(self, method, url, body, headers):
        return (httplib.NOT_FOUND, body, headers, httplib.responses[httplib.NOT_FOUND])

    def _foo_bar_container_NOT_FOUND(self, method, url, body, headers):
        return (httplib.NOT_FOUND, body, headers, httplib.responses[httplib.NOT_FOUND])

    def _foo_bar_container_foo_bar_object_NOT_FOUND(self, method, url, body, headers):
        return (httplib.NOT_FOUND, body, headers, httplib.responses[httplib.NOT_FOUND])

    def _foo_bar_container_foo_bar_object_DELETE(self, method, url, body, headers):
        return (httplib.ACCEPTED, body, headers, httplib.responses[httplib.ACCEPTED])

    def _foo_bar_container_foo_test_upload(self, method, url, body, headers):
        self._assert_content_length_header_is_string(headers=headers)
        query_string = urlparse.urlsplit(url).query
        query = parse_qs(query_string)
        comp = query.get('comp', [])
        headers = {}
        body = ''
        if 'blocklist' in comp or not comp:
            headers['etag'] = '"0x8CFB877BB56A6FB"'
            headers['content-md5'] = 'd4fe4c9829f7ca1cc89db7ad670d2bbd'
        elif 'block' in comp:
            headers['content-md5'] = 'lvcfx/bOJvndpRlrdKU1YQ=='
        else:
            raise NotImplementedError('Unknown request comp: {}'.format(comp))
        return (httplib.CREATED, body, headers, httplib.responses[httplib.CREATED])

    def _foo_bar_container_foo_test_upload_block(self, method, url, body, headers):
        self._assert_content_length_header_is_string(headers=headers)
        body = ''
        headers = {}
        headers['etag'] = '0x8CFB877BB56A6FB'
        return (httplib.CREATED, body, headers, httplib.responses[httplib.CREATED])

    def _foo_bar_container_foo_test_upload_blocklist(self, method, url, body, headers):
        self._assert_content_length_header_is_string(headers=headers)
        body = ''
        headers = {}
        headers['etag'] = '0x8CFB877BB56A6FB'
        headers['content-md5'] = 'd4fe4c9829f7ca1cc89db7ad670d2bbd'
        return (httplib.CREATED, body, headers, httplib.responses[httplib.CREATED])

    def _foo_bar_container_foo_test_upload_lease(self, method, url, body, headers):
        self._assert_content_length_header_is_string(headers=headers)
        action = headers['x-ms-lease-action']
        rheaders = {'x-ms-lease-id': 'someleaseid'}
        body = ''
        if action == 'acquire':
            return (httplib.CREATED, body, rheaders, httplib.responses[httplib.CREATED])
        else:
            if headers.get('x-ms-lease-id', None) != 'someleaseid':
                return (httplib.BAD_REQUEST, body, rheaders, httplib.responses[httplib.BAD_REQUEST])
            return (httplib.OK, body, headers, httplib.responses[httplib.CREATED])

    def _foo_bar_container_foo_test_upload_INVALID_HASH(self, method, url, body, headers):
        self._assert_content_length_header_is_string(headers=headers)
        body = ''
        headers = {}
        headers['etag'] = '0x8CFB877BB56A6FB'
        headers['content-md5'] = 'd4fe4c9829f7ca1cc89db7ad670d2bbd'
        return (httplib.CREATED, body, headers, httplib.responses[httplib.CREATED])

    def _foo_bar_container_foo_bar_object(self, method, url, body, headers):
        self._assert_content_length_header_is_string(headers=headers)
        body = generate_random_data(1000)
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _foo_bar_container_foo_bar_object_range(self, method, url, body, headers):
        body = '0123456789123456789'
        self.assertTrue('x-ms-range' in headers)
        self.assertEqual(headers['x-ms-range'], 'bytes=5-6')
        start_bytes, end_bytes = self._get_start_and_end_bytes_from_range_str(headers['x-ms-range'], body)
        return (httplib.PARTIAL_CONTENT, body[start_bytes:end_bytes + 1], headers, httplib.responses[httplib.PARTIAL_CONTENT])

    def _foo_bar_container_foo_bar_object_range_stream(self, method, url, body, headers):
        body = '0123456789123456789'
        self.assertTrue('x-ms-range' in headers)
        self.assertEqual(headers['x-ms-range'], 'bytes=4-5')
        start_bytes, end_bytes = self._get_start_and_end_bytes_from_range_str(headers['x-ms-range'], body)
        return (httplib.PARTIAL_CONTENT, body[start_bytes:end_bytes + 1], headers, httplib.responses[httplib.PARTIAL_CONTENT])

    def _foo_bar_container_foo_bar_object_INVALID_SIZE(self, method, url, body, headers):
        self._assert_content_length_header_is_string(headers=headers)
        body = ''
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _assert_content_length_header_is_string(self, headers):
        if 'Content-Length' in headers:
            self.assertTrue(isinstance(headers['Content-Length'], basestring))