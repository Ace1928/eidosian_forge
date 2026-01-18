import os
import sys
import json
import tempfile
from unittest import mock
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import b, httplib
from libcloud.utils.files import exhaust_iterator
from libcloud.test.file_fixtures import StorageFileFixtures
from libcloud.storage.drivers.backblaze_b2 import BackblazeB2StorageDriver
class BackblazeB2MockHttp(MockHttp):
    fixtures = StorageFileFixtures('backblaze_b2')

    def _b2api_v1_b2_authorize_account(self, method, url, body, headers):
        if method == 'GET':
            body = json.dumps({'accountId': 'test', 'apiUrl': 'https://apiNNN.backblazeb2.com', 'downloadUrl': 'https://f002.backblazeb2.com', 'authorizationToken': 'test'})
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _b2api_v1_b2_list_buckets(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('b2_list_buckets.json')
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _b2api_v1_b2_list_file_names(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('b2_list_file_names.json')
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _b2api_v1_b2_create_bucket(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('b2_create_bucket.json')
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _b2api_v1_b2_delete_bucket(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('b2_delete_bucket.json')
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _b2api_v1_b2_delete_file_version(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('b2_delete_file_version.json')
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _b2api_v1_b2_get_upload_url(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('b2_get_upload_url.json')
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _b2api_v1_b2_upload_file_abcd_defg(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('b2_upload_file.json')
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _b2api_v1_b2_list_file_versions(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('b2_list_file_versions.json')
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _b2api_v1_b2_hide_file(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('b2_hide_file.json')
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _file_test00001_2_txt(self, method, url, body, headers):
        if method == 'GET':
            body = 'ab'
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])