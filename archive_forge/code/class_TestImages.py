import hashlib
import http.client as http
import os
import subprocess
import tempfile
import time
import urllib
import uuid
import fixtures
from oslo_limit import exception as ol_exc
from oslo_limit import limit
from oslo_serialization import jsonutils
from oslo_utils.secretutils import md5
from oslo_utils import units
import requests
from glance.quota import keystone as ks_quota
from glance.tests import functional
from glance.tests.functional import ft_utils as func_utils
from glance.tests import utils as test_utils
class TestImages(functional.FunctionalTest):

    def setUp(self):
        super(TestImages, self).setUp()
        self.cleanup()
        self.include_scrubber = False
        self.api_server.deployment_flavor = 'noauth'
        for i in range(3):
            ret = test_utils.start_http_server('foo_image_id%d' % i, 'foo_image%d' % i)
            setattr(self, 'http_server%d' % i, ret[1])
            setattr(self, 'http_port%d' % i, ret[2])

    def tearDown(self):
        for i in range(3):
            httpd = getattr(self, 'http_server%d' % i, None)
            if httpd:
                httpd.shutdown()
                httpd.server_close()
        super(TestImages, self).tearDown()

    def _headers(self, custom_headers=None):
        base_headers = {'X-Identity-Status': 'Confirmed', 'X-Auth-Token': '932c5c84-02ac-4fe5-a9ba-620af0e2bb96', 'X-User-Id': 'f9a41d13-0c13-47e9-bee2-ce4e8bfe958e', 'X-Tenant-Id': TENANT1, 'X-Roles': 'reader,member'}
        base_headers.update(custom_headers or {})
        return base_headers

    def test_image_import_using_glance_direct(self):
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        path = self._url('/v2/info/import')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        discovery_calls = jsonutils.loads(response.text)['import-methods']['value']
        self.assertIn('glance-direct', discovery_calls)
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = jsonutils.dumps({'name': 'image-1', 'type': 'kernel', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        checked_keys = set(['status', 'name', 'tags', 'created_at', 'updated_at', 'visibility', 'self', 'protected', 'os_hidden', 'id', 'file', 'min_disk', 'type', 'min_ram', 'schema', 'disk_format', 'container_format', 'owner', 'checksum', 'os_hash_algo', 'os_hash_value', 'size', 'virtual_size'])
        self.assertEqual(checked_keys, set(image.keys()))
        expected_image = {'status': 'queued', 'name': 'image-1', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image_id, 'protected': False, 'file': '/v2/images/%s/file' % image_id, 'min_disk': 0, 'type': 'kernel', 'min_ram': 0, 'schema': '/v2/schemas/image'}
        for key, value in expected_image.items():
            self.assertEqual(value, image[key], key)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(1, len(images))
        self.assertEqual(image_id, images[0]['id'])
        path = self._url('/v2/images/%s/stage' % image_id)
        headers = self._headers({'Content-Type': 'application/octet-stream'})
        image_data = b'ZZZZZ'
        response = requests.put(path, headers=headers, data=image_data)
        self.assertEqual(http.NO_CONTENT, response.status_code)
        func_utils.verify_image_hashes_and_status(self, image_id, size=len(image_data), status='uploading')
        path = self._url('/v2/images/%s/import' % image_id)
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        data = jsonutils.dumps({'method': {'name': 'glance-direct'}})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.ACCEPTED, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        func_utils.wait_for_status(request_path=path, request_headers=self._headers(), status='active', max_sec=10, delay_sec=0.2)
        expect_c = str(md5(image_data, usedforsecurity=False).hexdigest())
        expect_h = str(hashlib.sha512(image_data).hexdigest())
        func_utils.verify_image_hashes_and_status(self, image_id, checksum=expect_c, os_hash_value=expect_h, size=len(image_data), status='active')
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        self.assertEqual(5, jsonutils.loads(response.text)['size'])
        path = self._url('/v2/images/%s' % image_id)
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        self.stop_servers()

    def test_image_import_using_web_download(self):
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        path = self._url('/v2/info/import')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        discovery_calls = jsonutils.loads(response.text)['import-methods']['value']
        self.assertIn('web-download', discovery_calls)
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = jsonutils.dumps({'name': 'image-1', 'type': 'kernel', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        checked_keys = set(['status', 'name', 'tags', 'created_at', 'updated_at', 'visibility', 'self', 'protected', 'os_hidden', 'id', 'file', 'min_disk', 'type', 'min_ram', 'schema', 'disk_format', 'container_format', 'owner', 'checksum', 'os_hash_algo', 'os_hash_value', 'size', 'virtual_size'])
        self.assertEqual(checked_keys, set(image.keys()))
        expected_image = {'status': 'queued', 'name': 'image-1', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image_id, 'protected': False, 'file': '/v2/images/%s/file' % image_id, 'min_disk': 0, 'type': 'kernel', 'min_ram': 0, 'schema': '/v2/schemas/image'}
        for key, value in expected_image.items():
            self.assertEqual(value, image[key], key)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(1, len(images))
        self.assertEqual(image_id, images[0]['id'])
        func_utils.verify_image_hashes_and_status(self, image_id, status='queued')
        path = self._url('/v2/images/%s/import' % image_id)
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        thread, httpd, port = test_utils.start_standalone_http_server()
        image_data_uri = 'http://localhost:%s/' % port
        data = jsonutils.dumps({'method': {'name': 'web-download', 'uri': image_data_uri}})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.ACCEPTED, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        func_utils.wait_for_status(request_path=path, request_headers=self._headers(), status='active', max_sec=20, delay_sec=0.2, start_delay_sec=1)
        with requests.get(image_data_uri) as r:
            expect_c = str(md5(r.content, usedforsecurity=False).hexdigest())
            expect_h = str(hashlib.sha512(r.content).hexdigest())
        func_utils.verify_image_hashes_and_status(self, image_id, checksum=expect_c, os_hash_value=expect_h, size=len(r.content), status='active')
        httpd.shutdown()
        httpd.server_close()
        path = self._url('/v2/images/%s' % image_id)
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        self.stop_servers()

    def test_image_lifecycle(self):
        self.api_server.show_multiple_locations = True
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = jsonutils.dumps({'name': 'image-1', 'type': 'kernel', 'foo': 'bar', 'disk_format': 'aki', 'container_format': 'aki', 'abc': 'xyz', 'protected': True})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image_location_header = response.headers['Location']
        image = jsonutils.loads(response.text)
        image_id = image['id']
        checked_keys = set(['status', 'name', 'tags', 'created_at', 'updated_at', 'visibility', 'self', 'protected', 'os_hidden', 'id', 'file', 'min_disk', 'foo', 'abc', 'type', 'min_ram', 'schema', 'disk_format', 'container_format', 'owner', 'checksum', 'os_hash_algo', 'os_hash_value', 'size', 'virtual_size', 'locations'])
        self.assertEqual(checked_keys, set(image.keys()))
        expected_image = {'status': 'queued', 'name': 'image-1', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image_id, 'protected': True, 'file': '/v2/images/%s/file' % image_id, 'min_disk': 0, 'foo': 'bar', 'abc': 'xyz', 'type': 'kernel', 'min_ram': 0, 'schema': '/v2/schemas/image'}
        for key, value in expected_image.items():
            self.assertEqual(value, image[key], key)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(1, len(images))
        self.assertEqual(image_id, images[0]['id'])
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = jsonutils.dumps({'name': 'image-2', 'type': 'kernel', 'bar': 'foo', 'disk_format': 'aki', 'container_format': 'aki', 'xyz': 'abc'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image2_id = image['id']
        checked_keys = set(['status', 'name', 'tags', 'created_at', 'updated_at', 'visibility', 'self', 'protected', 'os_hidden', 'id', 'file', 'min_disk', 'bar', 'xyz', 'type', 'min_ram', 'schema', 'disk_format', 'container_format', 'owner', 'checksum', 'os_hash_algo', 'os_hash_value', 'size', 'virtual_size', 'locations'])
        self.assertEqual(checked_keys, set(image.keys()))
        expected_image = {'status': 'queued', 'name': 'image-2', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image2_id, 'protected': False, 'file': '/v2/images/%s/file' % image2_id, 'min_disk': 0, 'bar': 'foo', 'xyz': 'abc', 'type': 'kernel', 'min_ram': 0, 'schema': '/v2/schemas/image'}
        for key, value in expected_image.items():
            self.assertEqual(value, image[key], key)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(2, len(images))
        self.assertEqual(image2_id, images[0]['id'])
        self.assertEqual(image_id, images[1]['id'])
        path = self._url('/v2/images?bar=foo')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(1, len(images))
        self.assertEqual(image2_id, images[0]['id'])
        path = self._url('/v2/images?foo=bar')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(1, len(images))
        self.assertEqual(image_id, images[0]['id'])
        path = self._url('/v2/images?changes-since=20001007T10:10:10')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.BAD_REQUEST, response.status_code)
        path = self._url('/v2/images?changes-since=aaa')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.BAD_REQUEST, response.status_code)
        path = self._url('/v2/images?protected=true')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(1, len(images))
        self.assertEqual(image_id, images[0]['id'])
        path = self._url('/v2/images?protected=false')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(1, len(images))
        self.assertEqual(image2_id, images[0]['id'])
        path = self._url('/v2/images?protected=False')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.BAD_REQUEST, response.status_code)
        path = self._url('/v2/images?foo=bar&abc=xyz')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(1, len(images))
        self.assertEqual(image_id, images[0]['id'])
        path = self._url('/v2/images?bar=foo&xyz=abc')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(1, len(images))
        self.assertEqual(image2_id, images[0]['id'])
        path = self._url('/v2/images?foo=baz&abc=xyz')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        response = requests.get(image_location_header, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        image = jsonutils.loads(response.text)
        self.assertEqual(image_id, image['id'])
        self.assertIsNone(image['checksum'])
        self.assertIsNone(image['size'])
        self.assertIsNone(image['virtual_size'])
        self.assertEqual('bar', image['foo'])
        self.assertTrue(image['protected'])
        self.assertEqual('kernel', image['type'])
        self.assertTrue(image['created_at'])
        self.assertTrue(image['updated_at'])
        self.assertEqual(image['updated_at'], image['created_at'])
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type})
        url = 'file://'
        changes = [{'op': 'add', 'path': '/locations/-', 'value': {'url': url, 'metadata': {}}}]
        data = jsonutils.dumps(changes)
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.BAD_REQUEST, response.status_code, response.text)
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type})
        data = jsonutils.dumps([{'op': 'replace', 'path': '/name', 'value': 'image-2'}, {'op': 'replace', 'path': '/disk_format', 'value': 'vhd'}, {'op': 'replace', 'path': '/container_format', 'value': 'ami'}, {'op': 'replace', 'path': '/foo', 'value': 'baz'}, {'op': 'add', 'path': '/ping', 'value': 'pong'}, {'op': 'replace', 'path': '/protected', 'value': True}, {'op': 'remove', 'path': '/type'}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.OK, response.status_code, response.text)
        image = jsonutils.loads(response.text)
        self.assertEqual('image-2', image['name'])
        self.assertEqual('vhd', image['disk_format'])
        self.assertEqual('baz', image['foo'])
        self.assertEqual('pong', image['ping'])
        self.assertTrue(image['protected'])
        self.assertNotIn('type', image, response.text)
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type})
        changes = []
        for i in range(11):
            changes.append({'op': 'add', 'path': '/ping%i' % i, 'value': 'pong'})
        data = jsonutils.dumps(changes)
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.REQUEST_ENTITY_TOO_LARGE, response.status_code, response.text)
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type})
        changes = []
        for i in range(3):
            url = 'http://127.0.0.1:%s/foo_image' % getattr(self, 'http_port%d' % i)
            changes.append({'op': 'add', 'path': '/locations/-', 'value': {'url': url, 'metadata': {}}})
        data = jsonutils.dumps(changes)
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.REQUEST_ENTITY_TOO_LARGE, response.status_code, response.text)
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.0-json-patch'
        headers = self._headers({'content-type': media_type})
        data = jsonutils.dumps([{'add': '/ding', 'value': 'dong'}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.OK, response.status_code, response.text)
        image = jsonutils.loads(response.text)
        self.assertEqual('dong', image['ding'])
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        image = jsonutils.loads(response.text)
        self.assertEqual(image_id, image['id'])
        self.assertEqual('image-2', image['name'])
        self.assertEqual('baz', image['foo'])
        self.assertEqual('pong', image['ping'])
        self.assertTrue(image['protected'])
        self.assertNotIn('type', image, response.text)
        path = self._url('/v2/images/%s/file' % image_id)
        headers = self._headers()
        response = requests.get(path, headers=headers)
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images/%s/file' % image_id)
        headers = self._headers({'Content-Type': 'application/octet-stream'})
        image_data = b'ZZZZZ'
        response = requests.put(path, headers=headers, data=image_data)
        self.assertEqual(http.NO_CONTENT, response.status_code)
        expect_c = str(md5(image_data, usedforsecurity=False).hexdigest())
        expect_h = str(hashlib.sha512(image_data).hexdigest())
        func_utils.verify_image_hashes_and_status(self, image_id, expect_c, expect_h, 'active', size=len(image_data))
        immutable_paths = ['/disk_format', '/container_format']
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type})
        path = self._url('/v2/images/%s' % image_id)
        for immutable_path in immutable_paths:
            data = jsonutils.dumps([{'op': 'replace', 'path': immutable_path, 'value': 'ari'}])
            response = requests.patch(path, headers=headers, data=data)
            self.assertEqual(http.FORBIDDEN, response.status_code)
        path = self._url('/v2/images/%s/file' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        self.assertEqual(expect_c, response.headers['Content-MD5'])
        self.assertEqual('ZZZZZ', response.text)
        path = self._url('/v2/images/%s/file' % image_id)
        headers = self._headers({'Content-Type': 'application/octet-stream'})
        response = requests.put(path, headers=headers, data='XXX')
        self.assertEqual(http.CONFLICT, response.status_code)
        func_utils.verify_image_hashes_and_status(self, image_id, expect_c, expect_h, 'active', size=len(image_data))
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        self.assertEqual(5, jsonutils.loads(response.text)['size'])
        path = self._url('/v2/images/%s/actions/deactivate' % image_id)
        response = requests.post(path, data={}, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.0-json-patch'
        headers = self._headers({'content-type': media_type, 'X-Roles': 'admin'})
        data = jsonutils.dumps([{'replace': '/visibility', 'value': 'public'}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.OK, response.status_code, response.text)
        path = self._url('/v2/images/%s/actions/deactivate' % image_id)
        response = requests.post(path, data={}, headers=self._headers({'X-Tenant-Id': TENANT2}))
        self.assertEqual(http.FORBIDDEN, response.status_code)
        path = self._url('/v2/images/%s/actions/reactivate' % image_id)
        response = requests.post(path, data={}, headers=self._headers({'X-Tenant-Id': TENANT2}))
        self.assertEqual(http.FORBIDDEN, response.status_code)
        path = self._url('/v2/images/%s/actions/deactivate' % image_id)
        response = requests.post(path, data={}, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images/%s/file' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.FORBIDDEN, response.status_code)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(2, len(images))
        self.assertEqual(image2_id, images[0]['id'])
        self.assertEqual(image_id, images[1]['id'])
        path = self._url('/v2/images/%s/actions/reactivate' % image_id)
        response = requests.post(path, data={}, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images/%s/actions/reactivate' % image_id)
        response = requests.post(path, data={}, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.FORBIDDEN, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type})
        doc = [{'op': 'replace', 'path': '/protected', 'value': False}]
        data = jsonutils.dumps(doc)
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.OK, response.status_code, response.text)
        path = self._url('/v2/images/%s' % image_id)
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.NOT_FOUND, response.status_code)
        path = self._url('/v2/images/%s/file' % image_id)
        headers = self._headers()
        response = requests.get(path, headers=headers)
        self.assertEqual(http.NOT_FOUND, response.status_code)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(1, len(images))
        self.assertEqual(image2_id, images[0]['id'])
        path = self._url('/v2/images/%s' % image2_id)
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = 'true'
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.BAD_REQUEST, response.status_code)
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = '"hello"'
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.BAD_REQUEST, response.status_code)
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = '123'
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.BAD_REQUEST, response.status_code)
        self.stop_servers()

    def _create_qcow(self, size):
        fn = tempfile.mktemp(prefix='glance-unittest-images-', suffix='.qcow2')
        subprocess.check_output('qemu-img create -f qcow2 %s %i' % (fn, size), shell=True)
        return fn

    def test_image_upload_qcow_virtual_size_calculation(self):
        self.start_servers(**self.__dict__.copy())
        headers = self._headers({'Content-Type': 'application/json'})
        data = jsonutils.dumps({'name': 'myqcow', 'disk_format': 'qcow2', 'container_format': 'bare'})
        response = requests.post(self._url('/v2/images'), headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code, 'Failed to create: %s' % response.text)
        image = response.json()
        fn = self._create_qcow(128 * units.Mi)
        raw_size = os.path.getsize(fn)
        headers = self._headers({'Content-Type': 'application/octet-stream'})
        response = requests.put(self._url('/v2/images/%s/file' % image['id']), headers=headers, data=open(fn, 'rb').read())
        os.remove(fn)
        self.assertEqual(http.NO_CONTENT, response.status_code)
        response = requests.get(self._url('/v2/images/%s' % image['id']), headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        image = response.json()
        self.assertEqual(128 * units.Mi, image['virtual_size'])
        self.assertEqual(raw_size, image['size'])

    def test_image_import_qcow_virtual_size_calculation(self):
        self.start_servers(**self.__dict__.copy())
        headers = self._headers({'Content-Type': 'application/json'})
        data = jsonutils.dumps({'name': 'myqcow', 'disk_format': 'qcow2', 'container_format': 'bare'})
        response = requests.post(self._url('/v2/images'), headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code, 'Failed to create: %s' % response.text)
        image = response.json()
        fn = self._create_qcow(128 * units.Mi)
        raw_size = os.path.getsize(fn)
        headers = self._headers({'Content-Type': 'application/octet-stream'})
        response = requests.put(self._url('/v2/images/%s/stage' % image['id']), headers=headers, data=open(fn, 'rb').read())
        os.remove(fn)
        self.assertEqual(http.NO_CONTENT, response.status_code)
        func_utils.verify_image_hashes_and_status(self, image['id'], status='uploading', size=raw_size)
        path = self._url('/v2/images/%s/import' % image['id'])
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        data = jsonutils.dumps({'method': {'name': 'glance-direct'}})
        response = requests.post(self._url('/v2/images/%s/import' % image['id']), headers=headers, data=data)
        self.assertEqual(http.ACCEPTED, response.status_code)
        path = self._url('/v2/images/%s' % image['id'])
        func_utils.wait_for_status(request_path=path, request_headers=self._headers(), status='active', max_sec=15, delay_sec=0.2)
        response = requests.get(self._url('/v2/images/%s' % image['id']), headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        image = response.json()
        self.assertEqual(128 * units.Mi, image['virtual_size'])
        self.assertEqual(raw_size, image['size'])

    def test_hidden_images(self):
        self.api_server.show_multiple_locations = True
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = jsonutils.dumps({'name': 'image-1', 'type': 'kernel', 'disk_format': 'aki', 'container_format': 'aki', 'protected': False})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        checked_keys = set(['status', 'name', 'tags', 'created_at', 'updated_at', 'visibility', 'self', 'protected', 'os_hidden', 'id', 'file', 'min_disk', 'type', 'min_ram', 'schema', 'disk_format', 'container_format', 'owner', 'checksum', 'os_hash_algo', 'os_hash_value', 'size', 'virtual_size', 'locations'])
        self.assertEqual(checked_keys, set(image.keys()))
        expected_image = {'status': 'queued', 'name': 'image-1', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image_id, 'protected': False, 'os_hidden': False, 'file': '/v2/images/%s/file' % image_id, 'min_disk': 0, 'type': 'kernel', 'min_ram': 0, 'schema': '/v2/schemas/image'}
        for key, value in expected_image.items():
            self.assertEqual(value, image[key], key)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(1, len(images))
        self.assertEqual(image_id, images[0]['id'])
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = jsonutils.dumps({'name': 'image-2', 'type': 'kernel', 'disk_format': 'aki', 'container_format': 'aki', 'os_hidden': True})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image2_id = image['id']
        checked_keys = set(['status', 'name', 'tags', 'created_at', 'updated_at', 'visibility', 'self', 'protected', 'os_hidden', 'id', 'file', 'min_disk', 'type', 'min_ram', 'schema', 'disk_format', 'container_format', 'owner', 'checksum', 'os_hash_algo', 'os_hash_value', 'size', 'virtual_size', 'locations'])
        self.assertEqual(checked_keys, set(image.keys()))
        expected_image = {'status': 'queued', 'name': 'image-2', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image2_id, 'protected': False, 'os_hidden': True, 'file': '/v2/images/%s/file' % image2_id, 'min_disk': 0, 'type': 'kernel', 'min_ram': 0, 'schema': '/v2/schemas/image'}
        for key, value in expected_image.items():
            self.assertEqual(value, image[key], key)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(1, len(images))
        self.assertEqual(image_id, images[0]['id'])
        path = self._url('/v2/images?os_hidden=false')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(1, len(images))
        self.assertEqual(image_id, images[0]['id'])
        path = self._url('/v2/images?os_hidden=true')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(1, len(images))
        self.assertEqual(image2_id, images[0]['id'])
        path = self._url('/v2/images?os_hidden=abcd')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.BAD_REQUEST, response.status_code)
        path = self._url('/v2/images/%s/file' % image_id)
        headers = self._headers({'Content-Type': 'application/octet-stream'})
        image_data = b'ZZZZZ'
        response = requests.put(path, headers=headers, data=image_data)
        self.assertEqual(http.NO_CONTENT, response.status_code)
        expect_c = str(md5(image_data, usedforsecurity=False).hexdigest())
        expect_h = str(hashlib.sha512(image_data).hexdigest())
        func_utils.verify_image_hashes_and_status(self, image_id, expect_c, expect_h, size=len(image_data), status='active')
        path = self._url('/v2/images/%s/file' % image2_id)
        headers = self._headers({'Content-Type': 'application/octet-stream'})
        image_data = b'WWWWW'
        response = requests.put(path, headers=headers, data=image_data)
        self.assertEqual(http.NO_CONTENT, response.status_code)
        expect_c = str(md5(image_data, usedforsecurity=False).hexdigest())
        expect_h = str(hashlib.sha512(image_data).hexdigest())
        func_utils.verify_image_hashes_and_status(self, image2_id, expect_c, expect_h, size=len(image_data), status='active')
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type})
        data = jsonutils.dumps([{'op': 'replace', 'path': '/os_hidden', 'value': True}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.OK, response.status_code, response.text)
        image = jsonutils.loads(response.text)
        self.assertTrue(image['os_hidden'])
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        path = self._url('/v2/images?os_hidden=true')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(2, len(images))
        self.assertEqual(image2_id, images[0]['id'])
        self.assertEqual(image_id, images[1]['id'])
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type})
        data = jsonutils.dumps([{'op': 'replace', 'path': '/os_hidden', 'value': False}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.OK, response.status_code, response.text)
        image = jsonutils.loads(response.text)
        self.assertFalse(image['os_hidden'])
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(1, len(images))
        self.assertEqual(image_id, images[0]['id'])
        path = self._url('/v2/images/%s' % image_id)
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images/%s' % image2_id)
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        self.stop_servers()

    def test_update_readonly_prop(self):
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = jsonutils.dumps({'name': 'image-1'})
        response = requests.post(path, headers=headers, data=data)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type})
        props = ['/id', '/file', '/location', '/schema', '/self']
        for prop in props:
            doc = [{'op': 'replace', 'path': prop, 'value': 'value1'}]
            data = jsonutils.dumps(doc)
            response = requests.patch(path, headers=headers, data=data)
            self.assertEqual(http.FORBIDDEN, response.status_code)
        for prop in props:
            doc = [{'op': 'remove', 'path': prop, 'value': 'value1'}]
            data = jsonutils.dumps(doc)
            response = requests.patch(path, headers=headers, data=data)
            self.assertEqual(http.FORBIDDEN, response.status_code)
        for prop in props:
            doc = [{'op': 'add', 'path': prop, 'value': 'value1'}]
            data = jsonutils.dumps(doc)
            response = requests.patch(path, headers=headers, data=data)
            self.assertEqual(http.FORBIDDEN, response.status_code)
        self.stop_servers()

    def test_methods_that_dont_accept_illegal_bodies(self):
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        schema_urls = ['/v2/schemas/images', '/v2/schemas/image', '/v2/schemas/members', '/v2/schemas/member']
        for value in schema_urls:
            path = self._url(value)
            data = jsonutils.dumps(['body'])
            response = requests.get(path, headers=self._headers(), data=data)
            self.assertEqual(http.BAD_REQUEST, response.status_code)
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = jsonutils.dumps({'name': 'image'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        test_urls = [('/v2/images/%s', 'get'), ('/v2/images/%s/actions/deactivate', 'post'), ('/v2/images/%s/actions/reactivate', 'post'), ('/v2/images/%s/tags/mytag', 'put'), ('/v2/images/%s/tags/mytag', 'delete'), ('/v2/images/%s/members', 'get'), ('/v2/images/%s/file', 'get'), ('/v2/images/%s', 'delete')]
        for link, method in test_urls:
            path = self._url(link % image_id)
            data = jsonutils.dumps(['body'])
            response = getattr(requests, method)(path, headers=self._headers(), data=data)
            self.assertEqual(http.BAD_REQUEST, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        data = '{"hello"]'
        response = requests.delete(path, headers=self._headers(), data=data)
        self.assertEqual(http.BAD_REQUEST, response.status_code)
        path = self._url('/v2/images/%s/members' % image_id)
        data = jsonutils.dumps({'member': TENANT3})
        response = requests.post(path, headers=self._headers(), data=data)
        self.assertEqual(http.OK, response.status_code)
        path = self._url('/v2/images/%s/members/%s' % (image_id, TENANT3))
        data = jsonutils.dumps(['body'])
        response = requests.get(path, headers=self._headers(), data=data)
        self.assertEqual(http.BAD_REQUEST, response.status_code)
        path = self._url('/v2/images/%s/members/%s' % (image_id, TENANT3))
        data = jsonutils.dumps(['body'])
        response = requests.delete(path, headers=self._headers(), data=data)
        self.assertEqual(http.BAD_REQUEST, response.status_code)
        self.stop_servers()

    def test_download_random_access_w_range_request(self):
        """
        Test partial download 'Range' requests for images (random image access)
        """
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = jsonutils.dumps({'name': 'image-2', 'type': 'kernel', 'bar': 'foo', 'disk_format': 'aki', 'container_format': 'aki', 'xyz': 'abc'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        image_data = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        path = self._url('/v2/images/%s/file' % image_id)
        headers = self._headers({'Content-Type': 'application/octet-stream'})
        response = requests.put(path, headers=headers, data=image_data)
        self.assertEqual(http.NO_CONTENT, response.status_code)
        range_ = 'bytes=3-10'
        headers = self._headers({'Range': range_})
        path = self._url('/v2/images/%s/file' % image_id)
        response = requests.get(path, headers=headers)
        self.assertEqual(http.PARTIAL_CONTENT, response.status_code)
        self.assertEqual('DEFGHIJK', response.text)
        range_ = 'bytes=10-5'
        headers = self._headers({'Range': range_})
        path = self._url('/v2/images/%s/file' % image_id)
        response = requests.get(path, headers=headers)
        self.assertEqual(http.REQUESTED_RANGE_NOT_SATISFIABLE, response.status_code)
        self.stop_servers()

    def test_download_random_access_w_content_range(self):
        """
        Even though Content-Range is incorrect on requests, we support it
        for backward compatibility with clients written for pre-Pike Glance.
        The following test is for 'Content-Range' requests, which we have
        to ensure that we prevent regression.
        """
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = jsonutils.dumps({'name': 'image-2', 'type': 'kernel', 'bar': 'foo', 'disk_format': 'aki', 'container_format': 'aki', 'xyz': 'abc'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        image_data = 'Z' * 15
        path = self._url('/v2/images/%s/file' % image_id)
        headers = self._headers({'Content-Type': 'application/octet-stream'})
        response = requests.put(path, headers=headers, data=image_data)
        self.assertEqual(http.NO_CONTENT, response.status_code)
        result_body = ''
        for x in range(15):
            content_range = 'bytes %s-%s/15' % (x, x)
            headers = self._headers({'Content-Range': content_range})
            path = self._url('/v2/images/%s/file' % image_id)
            response = requests.get(path, headers=headers)
            self.assertEqual(http.PARTIAL_CONTENT, response.status_code)
            result_body += response.text
        self.assertEqual(result_body, image_data)
        content_range = 'bytes 3-16/15'
        headers = self._headers({'Content-Range': content_range})
        path = self._url('/v2/images/%s/file' % image_id)
        response = requests.get(path, headers=headers)
        self.assertEqual(http.REQUESTED_RANGE_NOT_SATISFIABLE, response.status_code)
        self.stop_servers()

    def test_download_policy_when_cache_is_not_enabled(self):
        rules = {'context_is_admin': 'role:admin', 'default': '', 'add_image': '', 'get_image': '', 'modify_image': '', 'upload_image': '', 'delete_image': '', 'download_image': '!'}
        self.set_policy_rules(rules)
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        expected_image = {'status': 'queued', 'name': 'image-1', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image_id, 'protected': False, 'file': '/v2/images/%s/file' % image_id, 'min_disk': 0, 'min_ram': 0, 'schema': '/v2/schemas/image'}
        for key, value in expected_image.items():
            self.assertEqual(value, image[key], key)
        path = self._url('/v2/images/%s/file' % image_id)
        headers = self._headers({'Content-Type': 'application/octet-stream'})
        response = requests.put(path, headers=headers, data='ZZZZZ')
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images/%s/file' % image_id)
        headers = self._headers({'Content-Type': 'application/octet-stream'})
        response = requests.get(path, headers=headers)
        self.assertEqual(http.FORBIDDEN, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.NOT_FOUND, response.status_code)
        self.stop_servers()

    def test_download_image_not_allowed_using_restricted_policy(self):
        rules = {'context_is_admin': 'role:admin', 'default': '', 'add_image': '', 'get_image': '', 'modify_image': '', 'upload_image': '', 'delete_image': '', 'restricted': "not ('aki':%(container_format)s and role:_member_)", 'download_image': 'role:admin or rule:restricted'}
        self.set_policy_rules(rules)
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        expected_image = {'status': 'queued', 'name': 'image-1', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image_id, 'protected': False, 'file': '/v2/images/%s/file' % image_id, 'min_disk': 0, 'min_ram': 0, 'schema': '/v2/schemas/image'}
        for key, value in expected_image.items():
            self.assertEqual(value, image[key], key)
        path = self._url('/v2/images/%s/file' % image_id)
        headers = self._headers({'Content-Type': 'application/octet-stream'})
        response = requests.put(path, headers=headers, data='ZZZZZ')
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images/%s/file' % image_id)
        headers = self._headers({'Content-Type': 'application/octet-stream', 'X-Roles': '_member_'})
        response = requests.get(path, headers=headers)
        self.assertEqual(http.FORBIDDEN, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.NOT_FOUND, response.status_code)
        self.stop_servers()

    def test_download_image_allowed_using_restricted_policy(self):
        rules = {'context_is_admin': 'role:admin', 'default': '', 'add_image': '', 'get_image': '', 'modify_image': '', 'upload_image': '', 'get_image_location': '', 'delete_image': '', 'restricted': "not ('aki':%(container_format)s and role:_member_)", 'download_image': 'role:admin or rule:restricted'}
        self.set_policy_rules(rules)
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        expected_image = {'status': 'queued', 'name': 'image-1', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image_id, 'protected': False, 'file': '/v2/images/%s/file' % image_id, 'min_disk': 0, 'min_ram': 0, 'schema': '/v2/schemas/image'}
        for key, value in expected_image.items():
            self.assertEqual(value, image[key], key)
        path = self._url('/v2/images/%s/file' % image_id)
        headers = self._headers({'Content-Type': 'application/octet-stream'})
        response = requests.put(path, headers=headers, data='ZZZZZ')
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images/%s/file' % image_id)
        headers = self._headers({'Content-Type': 'application/octet-stream', 'X-Roles': 'reader,member'})
        response = requests.get(path, headers=headers)
        self.assertEqual(http.OK, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.NOT_FOUND, response.status_code)
        self.stop_servers()

    def test_download_image_raises_service_unavailable(self):
        """Test image download returns HTTPServiceUnavailable."""
        self.api_server.show_multiple_locations = True
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type})
        thread, httpd, http_port = test_utils.start_http_server(image_id, 'image-1')
        values = [{'url': 'http://127.0.0.1:%s/image-1' % http_port, 'metadata': {'idx': '0'}}]
        doc = [{'op': 'replace', 'path': '/locations', 'value': values}]
        data = jsonutils.dumps(doc)
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.OK, response.status_code)
        path = self._url('/v2/images/%s/file' % image_id)
        headers = self._headers({'Content-Type': 'application/json'})
        response = requests.get(path, headers=headers)
        self.assertEqual(http.OK, response.status_code)
        httpd.shutdown()
        httpd.server_close()
        path = self._url('/v2/images/%s/file' % image_id)
        headers = self._headers({'Content-Type': 'application/json'})
        response = requests.get(path, headers=headers)
        self.assertEqual(http.SERVICE_UNAVAILABLE, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.NOT_FOUND, response.status_code)
        self.stop_servers()

    def test_image_modification_works_for_owning_tenant_id(self):
        rules = {'context_is_admin': 'role:admin', 'default': '', 'add_image': '', 'get_image': '', 'modify_image': 'project_id:%(owner)s', 'upload_image': '', 'get_image_location': '', 'delete_image': '', 'restricted': "not ('aki':%(container_format)s and role:_member_)", 'download_image': 'role:admin or rule:restricted'}
        self.set_policy_rules(rules)
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers['content-type'] = media_type
        del headers['X-Roles']
        data = jsonutils.dumps([{'op': 'replace', 'path': '/name', 'value': 'new-name'}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.OK, response.status_code)
        self.stop_servers()

    def test_image_modification_fails_on_mismatched_tenant_ids(self):
        rules = {'context_is_admin': 'role:admin', 'default': '', 'add_image': '', 'get_image': '', 'modify_image': "'A-Fake-Tenant-Id':%(owner)s", 'upload_image': '', 'get_image_location': '', 'delete_image': '', 'restricted': "not ('aki':%(container_format)s and role:_member_)", 'download_image': 'role:admin or rule:restricted'}
        self.set_policy_rules(rules)
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers['content-type'] = media_type
        del headers['X-Roles']
        data = jsonutils.dumps([{'op': 'replace', 'path': '/name', 'value': 'new-name'}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.FORBIDDEN, response.status_code)
        self.stop_servers()

    def test_member_additions_works_for_owning_tenant_id(self):
        rules = {'context_is_admin': 'role:admin', 'default': '', 'add_image': '', 'get_image': '', 'modify_image': '', 'upload_image': '', 'get_image_location': '', 'delete_image': '', 'restricted': "not ('aki':%(container_format)s and role:_member_)", 'download_image': 'role:admin or rule:restricted', 'add_member': 'project_id:%(owner)s'}
        self.set_policy_rules(rules)
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        path = self._url('/v2/images/%s/members' % image_id)
        body = jsonutils.dumps({'member': TENANT3})
        del headers['X-Roles']
        response = requests.post(path, headers=headers, data=body)
        self.assertEqual(http.OK, response.status_code)
        self.stop_servers()

    def test_image_additions_works_only_for_specific_tenant_id(self):
        rules = {'context_is_admin': 'role:admin', 'default': '', 'add_image': "'{0}':%(owner)s".format(TENANT1), 'get_image': '', 'modify_image': '', 'upload_image': '', 'get_image_location': '', 'delete_image': '', 'restricted': "not ('aki':%(container_format)s and role:_member_)", 'download_image': 'role:admin or rule:restricted', 'add_member': ''}
        self.set_policy_rules(rules)
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin', 'X-Tenant-Id': TENANT1})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        headers['X-Tenant-Id'] = TENANT2
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.FORBIDDEN, response.status_code)
        self.stop_servers()

    def test_owning_tenant_id_can_retrieve_image_information(self):
        rules = {'context_is_admin': 'role:admin', 'default': '', 'add_image': '', 'get_image': 'project_id:%(owner)s', 'modify_image': '', 'upload_image': '', 'get_image_location': '', 'delete_image': '', 'restricted': "not ('aki':%(container_format)s and role:_member_)", 'download_image': 'role:admin or rule:restricted', 'add_member': ''}
        self.set_policy_rules(rules)
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin', 'X-Tenant-Id': TENANT1})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        headers['X-Roles'] = 'reader,member'
        image = jsonutils.loads(response.text)
        image_id = image['id']
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=headers)
        self.assertEqual(http.OK, response.status_code)
        path = self._url('/v2/images/%s/members' % image_id)
        response = requests.get(path, headers=headers)
        self.assertEqual(http.OK, response.status_code)
        headers['X-Tenant-Id'] = TENANT2
        response = requests.get(path, headers=headers)
        self.assertEqual(http.NOT_FOUND, response.status_code)
        self.stop_servers()

    def test_owning_tenant_can_publicize_image(self):
        rules = {'context_is_admin': 'role:admin', 'default': '', 'add_image': '', 'publicize_image': 'project_id:%(owner)s', 'get_image': 'project_id:%(owner)s', 'modify_image': '', 'upload_image': '', 'get_image_location': '', 'delete_image': '', 'restricted': "not ('aki':%(container_format)s and role:_member_)", 'download_image': 'role:admin or rule:restricted', 'add_member': ''}
        self.set_policy_rules(rules)
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin', 'X-Tenant-Id': TENANT1})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        path = self._url('/v2/images/%s' % image_id)
        headers = self._headers({'Content-Type': 'application/openstack-images-v2.1-json-patch', 'X-Tenant-Id': TENANT1})
        doc = [{'op': 'replace', 'path': '/visibility', 'value': 'public'}]
        data = jsonutils.dumps(doc)
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.OK, response.status_code)

    def test_owning_tenant_can_communitize_image(self):
        rules = {'context_is_admin': 'role:admin', 'default': '', 'add_image': '', 'communitize_image': 'project_id:%(owner)s', 'get_image': 'project_id:%(owner)s', 'modify_image': '', 'upload_image': '', 'get_image_location': '', 'delete_image': '', 'restricted': "not ('aki':%(container_format)s and role:_member_)", 'download_image': 'role:admin or rule:restricted', 'add_member': ''}
        self.set_policy_rules(rules)
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin', 'X-Tenant-Id': TENANT1})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(201, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        path = self._url('/v2/images/%s' % image_id)
        headers = self._headers({'Content-Type': 'application/openstack-images-v2.1-json-patch', 'X-Tenant-Id': TENANT1})
        doc = [{'op': 'replace', 'path': '/visibility', 'value': 'community'}]
        data = jsonutils.dumps(doc)
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(200, response.status_code)

    def test_owning_tenant_can_delete_image(self):
        rules = {'context_is_admin': 'role:admin', 'default': '', 'add_image': '', 'publicize_image': 'project_id:%(owner)s', 'get_image': 'project_id:%(owner)s', 'modify_image': '', 'upload_image': '', 'get_image_location': '', 'delete_image': '', 'restricted': "not ('aki':%(container_format)s and role:_member_)", 'download_image': 'role:admin or rule:restricted', 'add_member': ''}
        self.set_policy_rules(rules)
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin', 'X-Tenant-Id': TENANT1})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        path = self._url('/v2/images/%s' % image_id)
        response = requests.delete(path, headers=headers)
        self.assertEqual(http.NO_CONTENT, response.status_code)

    def test_list_show_ok_when_get_location_allowed_for_admins(self):
        self.api_server.show_image_direct_url = True
        self.api_server.show_multiple_locations = True
        rules = {'context_is_admin': 'role:admin', 'default': '', 'add_image': '', 'get_image': '', 'modify_image': '', 'upload_image': '', 'get_image_location': 'role:admin', 'delete_image': '', 'restricted': '', 'download_image': '', 'add_member': ''}
        self.set_policy_rules(rules)
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Tenant-Id': TENANT1})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=headers)
        self.assertEqual(http.OK, response.status_code)
        path = self._url('/v2/images')
        response = requests.get(path, headers=headers)
        self.assertEqual(http.OK, response.status_code)
        self.stop_servers()

    def test_image_size_cap(self):
        self.api_server.image_size_cap = 128
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = jsonutils.dumps({'name': 'image-size-cap-test-image', 'type': 'kernel', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        path = self._url('/v2/images/%s/file' % image_id)
        headers = self._headers({'Content-Type': 'application/octet-stream'})

        class StreamSim(object):

            def __init__(self, size):
                self.size = size

            def __iter__(self):
                yield (b'Z' * self.size)
        response = requests.put(path, headers=headers, data=StreamSim(self.api_server.image_size_cap + 1))
        self.assertEqual(http.REQUEST_ENTITY_TOO_LARGE, response.status_code)
        path = self._url('/v2/images/{0}'.format(image_id))
        headers = self._headers({'content-type': 'application/json'})
        response = requests.get(path, headers=headers)
        image_checksum = jsonutils.loads(response.text).get('checksum')
        self.assertNotEqual(image_checksum, '76522d28cb4418f12704dfa7acd6e7ee')

    def test_permissions(self):
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        headers = self._headers({'Content-Type': 'application/json'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'raw', 'container_format': 'bare'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image_id = jsonutils.loads(response.text)['id']
        path = self._url('/v2/images/%s/file' % image_id)
        headers = self._headers({'Content-Type': 'application/octet-stream'})
        response = requests.put(path, headers=headers, data='ZZZZZ')
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(image_id, images[0]['id'])
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        path = self._url('/v2/images')
        headers = self._headers({'X-Tenant-Id': TENANT2})
        response = requests.get(path, headers=headers)
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        path = self._url('/v2/images/%s' % image_id)
        headers = self._headers({'X-Tenant-Id': TENANT2})
        response = requests.get(path, headers=headers)
        self.assertEqual(http.NOT_FOUND, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        headers = self._headers({'Content-Type': 'application/openstack-images-v2.1-json-patch', 'X-Tenant-Id': TENANT2})
        doc = [{'op': 'replace', 'path': '/name', 'value': 'image-2'}]
        data = jsonutils.dumps(doc)
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.NOT_FOUND, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        headers = self._headers({'X-Tenant-Id': TENANT2})
        response = requests.delete(path, headers=headers)
        self.assertEqual(http.NOT_FOUND, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        headers = self._headers({'Content-Type': 'application/openstack-images-v2.1-json-patch', 'X-Roles': 'admin'})
        doc = [{'op': 'replace', 'path': '/visibility', 'value': 'public'}]
        data = jsonutils.dumps(doc)
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.OK, response.status_code)
        path = self._url('/v2/images')
        headers = self._headers({'X-Tenant-Id': TENANT3})
        response = requests.get(path, headers=headers)
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(image_id, images[0]['id'])
        path = self._url('/v2/images/%s' % image_id)
        headers = self._headers({'X-Tenant-Id': TENANT3})
        response = requests.get(path, headers=headers)
        self.assertEqual(http.OK, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        headers = self._headers({'Content-Type': 'application/openstack-images-v2.1-json-patch', 'X-Tenant-Id': TENANT3})
        doc = [{'op': 'replace', 'path': '/name', 'value': 'image-2'}]
        data = jsonutils.dumps(doc)
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.FORBIDDEN, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        headers = self._headers({'X-Tenant-Id': TENANT3})
        response = requests.delete(path, headers=headers)
        self.assertEqual(http.FORBIDDEN, response.status_code)
        path = self._url('/v2/images/%s/file' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        self.assertEqual(response.text, 'ZZZZZ')
        self.stop_servers()

    def test_property_protections_with_roles(self):
        self.api_server.property_protection_file = self.property_file_roles
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member'})
        data = jsonutils.dumps({'name': 'image-1', 'foo': 'bar', 'disk_format': 'aki', 'container_format': 'aki', 'x_owner_foo': 'o_s_bar'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.FORBIDDEN, response.status_code)
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'x_owner_foo': 'o_s_bar'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        expected_image = {'status': 'queued', 'name': 'image-1', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image_id, 'protected': False, 'file': '/v2/images/%s/file' % image_id, 'min_disk': 0, 'x_owner_foo': 'o_s_bar', 'min_ram': 0, 'schema': '/v2/schemas/image'}
        for key, value in expected_image.items():
            self.assertEqual(value, image[key], key)
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member,spl_role'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'spl_create_prop': 'create_bar', 'spl_create_prop_policy': 'create_policy_bar', 'spl_read_prop': 'read_bar', 'spl_update_prop': 'update_bar', 'spl_delete_prop': 'delete_bar', 'spl_delete_empty_prop': ''})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type, 'X-Roles': 'reader,member,spl_role'})
        data = jsonutils.dumps([{'op': 'replace', 'path': '/spl_read_prop', 'value': 'r'}, {'op': 'replace', 'path': '/spl_update_prop', 'value': 'u'}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.FORBIDDEN, response.status_code, response.text)
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type, 'X-Roles': 'reader,member,spl_role'})
        data = jsonutils.dumps([{'op': 'add', 'path': '/spl_new_prop', 'value': 'new'}, {'op': 'remove', 'path': '/spl_create_prop'}, {'op': 'remove', 'path': '/spl_delete_prop'}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.FORBIDDEN, response.status_code, response.text)
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type, 'X-Roles': 'reader,member,spl_role'})
        data = jsonutils.dumps([{'op': 'replace', 'path': '/spl_update_prop', 'value': ''}, {'op': 'replace', 'path': '/spl_update_prop', 'value': 'u'}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.OK, response.status_code, response.text)
        image = jsonutils.loads(response.text)
        self.assertEqual('u', image['spl_update_prop'])
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type, 'X-Roles': 'reader,member,spl_role'})
        data = jsonutils.dumps([{'op': 'remove', 'path': '/spl_delete_prop'}, {'op': 'remove', 'path': '/spl_delete_empty_prop'}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.OK, response.status_code, response.text)
        image = jsonutils.loads(response.text)
        self.assertNotIn('spl_delete_prop', image.keys())
        self.assertNotIn('spl_delete_empty_prop', image.keys())
        path = self._url('/v2/images/%s' % image_id)
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.NOT_FOUND, response.status_code)
        self.stop_servers()

    def test_property_protections_with_policies(self):
        rules = {'glance_creator': 'role:admin or role:spl_role'}
        self.set_policy_rules(rules)
        self.api_server.property_protection_file = self.property_file_policies
        self.api_server.property_protection_rule_format = 'policies'
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member'})
        data = jsonutils.dumps({'name': 'image-1', 'foo': 'bar', 'disk_format': 'aki', 'container_format': 'aki', 'x_owner_foo': 'o_s_bar'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.FORBIDDEN, response.status_code)
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        expected_image = {'status': 'queued', 'name': 'image-1', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image_id, 'protected': False, 'file': '/v2/images/%s/file' % image_id, 'min_disk': 0, 'min_ram': 0, 'schema': '/v2/schemas/image'}
        for key, value in expected_image.items():
            self.assertEqual(value, image[key], key)
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member,spl_role, admin'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'spl_creator_policy': 'creator_bar', 'spl_default_policy': 'default_bar'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        self.assertEqual('creator_bar', image['spl_creator_policy'])
        self.assertEqual('default_bar', image['spl_default_policy'])
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type, 'X-Roles': 'admin'})
        data = jsonutils.dumps([{'op': 'replace', 'path': '/spl_creator_policy', 'value': ''}, {'op': 'replace', 'path': '/spl_creator_policy', 'value': 'r'}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.OK, response.status_code, response.text)
        image = jsonutils.loads(response.text)
        self.assertEqual('r', image['spl_creator_policy'])
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type, 'X-Roles': 'reader,member,spl_role'})
        data = jsonutils.dumps([{'op': 'replace', 'path': '/spl_creator_policy', 'value': 'z'}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.FORBIDDEN, response.status_code, response.text)
        path = self._url('/v2/images/%s' % image_id)
        headers = self._headers({'content-type': media_type, 'X-Roles': 'reader,random_role'})
        response = requests.get(path, headers=headers)
        self.assertEqual(http.OK, response.status_code)
        image = jsonutils.loads(response.text)
        self.assertEqual(image['spl_default_policy'], 'default_bar')
        self.assertNotIn('spl_creator_policy', image)
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type, 'X-Roles': 'admin'})
        data = jsonutils.dumps([{'op': 'replace', 'path': '/spl_creator_policy', 'value': ''}, {'op': 'remove', 'path': '/spl_creator_policy'}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.OK, response.status_code, response.text)
        image = jsonutils.loads(response.text)
        self.assertNotIn('spl_creator_policy', image)
        path = self._url('/v2/images/%s' % image_id)
        headers = self._headers({'content-type': media_type, 'X-Roles': 'reader,random_role'})
        response = requests.get(path, headers=headers)
        self.assertEqual(http.OK, response.status_code)
        image = jsonutils.loads(response.text)
        self.assertEqual(image['spl_default_policy'], 'default_bar')
        path = self._url('/v2/images/%s' % image_id)
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.NOT_FOUND, response.status_code)
        self.stop_servers()

    def test_property_protections_special_chars_roles(self):
        self.api_server.property_protection_file = self.property_file_roles
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'x_all_permitted_admin': '1'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        expected_image = {'status': 'queued', 'name': 'image-1', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image_id, 'protected': False, 'file': '/v2/images/%s/file' % image_id, 'min_disk': 0, 'x_all_permitted_admin': '1', 'min_ram': 0, 'schema': '/v2/schemas/image'}
        for key, value in expected_image.items():
            self.assertEqual(value, image[key], key)
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member,joe_soap'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'x_all_permitted_joe_soap': '1'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        expected_image = {'status': 'queued', 'name': 'image-1', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image_id, 'protected': False, 'file': '/v2/images/%s/file' % image_id, 'min_disk': 0, 'x_all_permitted_joe_soap': '1', 'min_ram': 0, 'schema': '/v2/schemas/image'}
        for key, value in expected_image.items():
            self.assertEqual(value, image[key], key)
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        image = jsonutils.loads(response.text)
        self.assertEqual('1', image['x_all_permitted_joe_soap'])
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member,joe_soap'})
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        image = jsonutils.loads(response.text)
        self.assertEqual('1', image['x_all_permitted_joe_soap'])
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type, 'X-Roles': 'admin'})
        data = jsonutils.dumps([{'op': 'replace', 'path': '/x_all_permitted_joe_soap', 'value': '2'}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.OK, response.status_code, response.text)
        image = jsonutils.loads(response.text)
        self.assertEqual('2', image['x_all_permitted_joe_soap'])
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type, 'X-Roles': 'reader,member,joe_soap'})
        data = jsonutils.dumps([{'op': 'replace', 'path': '/x_all_permitted_joe_soap', 'value': '3'}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.OK, response.status_code, response.text)
        image = jsonutils.loads(response.text)
        self.assertEqual('3', image['x_all_permitted_joe_soap'])
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'x_all_permitted_a': '1', 'x_all_permitted_b': '2'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type, 'X-Roles': 'admin'})
        data = jsonutils.dumps([{'op': 'remove', 'path': '/x_all_permitted_a'}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.OK, response.status_code, response.text)
        image = jsonutils.loads(response.text)
        self.assertNotIn('x_all_permitted_a', image.keys())
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type, 'X-Roles': 'reader,member,joe_soap'})
        data = jsonutils.dumps([{'op': 'remove', 'path': '/x_all_permitted_b'}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.OK, response.status_code, response.text)
        image = jsonutils.loads(response.text)
        self.assertNotIn('x_all_permitted_b', image.keys())
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'x_none_permitted_admin': '1'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.FORBIDDEN, response.status_code)
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member,joe_soap'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'x_none_permitted_joe_soap': '1'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.FORBIDDEN, response.status_code)
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'x_none_read': '1'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        self.assertNotIn('x_none_read', image.keys())
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        image = jsonutils.loads(response.text)
        self.assertNotIn('x_none_read', image.keys())
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member,joe_soap'})
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        image = jsonutils.loads(response.text)
        self.assertNotIn('x_none_read', image.keys())
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'x_none_update': '1'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        self.assertEqual('1', image['x_none_update'])
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type, 'X-Roles': 'admin'})
        data = jsonutils.dumps([{'op': 'replace', 'path': '/x_none_update', 'value': '2'}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.FORBIDDEN, response.status_code, response.text)
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type, 'X-Roles': 'reader,member,joe_soap'})
        data = jsonutils.dumps([{'op': 'replace', 'path': '/x_none_update', 'value': '3'}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.FORBIDDEN, response.status_code, response.text)
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'x_none_delete': '1'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type, 'X-Roles': 'admin'})
        data = jsonutils.dumps([{'op': 'remove', 'path': '/x_none_delete'}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.FORBIDDEN, response.status_code, response.text)
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type, 'X-Roles': 'reader,member,joe_soap'})
        data = jsonutils.dumps([{'op': 'remove', 'path': '/x_none_delete'}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.FORBIDDEN, response.status_code, response.text)
        self.stop_servers()

    def test_property_protections_special_chars_policies(self):
        self.api_server.property_protection_file = self.property_file_policies
        self.api_server.property_protection_rule_format = 'policies'
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'x_all_permitted_admin': '1'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        expected_image = {'status': 'queued', 'name': 'image-1', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image_id, 'protected': False, 'file': '/v2/images/%s/file' % image_id, 'min_disk': 0, 'x_all_permitted_admin': '1', 'min_ram': 0, 'schema': '/v2/schemas/image'}
        for key, value in expected_image.items():
            self.assertEqual(value, image[key], key)
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member,joe_soap'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'x_all_permitted_joe_soap': '1'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        expected_image = {'status': 'queued', 'name': 'image-1', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image_id, 'protected': False, 'file': '/v2/images/%s/file' % image_id, 'min_disk': 0, 'x_all_permitted_joe_soap': '1', 'min_ram': 0, 'schema': '/v2/schemas/image'}
        for key, value in expected_image.items():
            self.assertEqual(value, image[key], key)
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        image = jsonutils.loads(response.text)
        self.assertEqual('1', image['x_all_permitted_joe_soap'])
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member,joe_soap'})
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        image = jsonutils.loads(response.text)
        self.assertEqual('1', image['x_all_permitted_joe_soap'])
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type, 'X-Roles': 'admin'})
        data = jsonutils.dumps([{'op': 'replace', 'path': '/x_all_permitted_joe_soap', 'value': '2'}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.OK, response.status_code, response.text)
        image = jsonutils.loads(response.text)
        self.assertEqual('2', image['x_all_permitted_joe_soap'])
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type, 'X-Roles': 'reader,member,joe_soap'})
        data = jsonutils.dumps([{'op': 'replace', 'path': '/x_all_permitted_joe_soap', 'value': '3'}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.OK, response.status_code, response.text)
        image = jsonutils.loads(response.text)
        self.assertEqual('3', image['x_all_permitted_joe_soap'])
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'x_all_permitted_a': '1', 'x_all_permitted_b': '2'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type, 'X-Roles': 'admin'})
        data = jsonutils.dumps([{'op': 'remove', 'path': '/x_all_permitted_a'}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.OK, response.status_code, response.text)
        image = jsonutils.loads(response.text)
        self.assertNotIn('x_all_permitted_a', image.keys())
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type, 'X-Roles': 'reader,member,joe_soap'})
        data = jsonutils.dumps([{'op': 'remove', 'path': '/x_all_permitted_b'}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.OK, response.status_code, response.text)
        image = jsonutils.loads(response.text)
        self.assertNotIn('x_all_permitted_b', image.keys())
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'x_none_permitted_admin': '1'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.FORBIDDEN, response.status_code)
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member,joe_soap'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'x_none_permitted_joe_soap': '1'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.FORBIDDEN, response.status_code)
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'x_none_read': '1'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        self.assertNotIn('x_none_read', image.keys())
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        image = jsonutils.loads(response.text)
        self.assertNotIn('x_none_read', image.keys())
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member,joe_soap'})
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        image = jsonutils.loads(response.text)
        self.assertNotIn('x_none_read', image.keys())
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'x_none_update': '1'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        self.assertEqual('1', image['x_none_update'])
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type, 'X-Roles': 'admin'})
        data = jsonutils.dumps([{'op': 'replace', 'path': '/x_none_update', 'value': '2'}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.FORBIDDEN, response.status_code, response.text)
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type, 'X-Roles': 'reader,member,joe_soap'})
        data = jsonutils.dumps([{'op': 'replace', 'path': '/x_none_update', 'value': '3'}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.CONFLICT, response.status_code, response.text)
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'x_none_delete': '1'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type, 'X-Roles': 'admin'})
        data = jsonutils.dumps([{'op': 'remove', 'path': '/x_none_delete'}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.FORBIDDEN, response.status_code, response.text)
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type, 'X-Roles': 'reader,member,joe_soap'})
        data = jsonutils.dumps([{'op': 'remove', 'path': '/x_none_delete'}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.CONFLICT, response.status_code, response.text)
        self.stop_servers()

    def test_tag_lifecycle(self):
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        headers = self._headers({'Content-Type': 'application/json'})
        data = jsonutils.dumps({'name': 'image-1', 'tags': ['sniff', 'sniff']})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image_id = jsonutils.loads(response.text)['id']
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        tags = jsonutils.loads(response.text)['tags']
        self.assertEqual(['sniff'], tags)
        for tag in tags:
            path = self._url('/v2/images/%s/tags/%s' % (image_id, tag))
            response = requests.delete(path, headers=self._headers())
            self.assertEqual(http.NO_CONTENT, response.status_code)
        for i in range(10):
            path = self._url('/v2/images/%s/tags/foo%i' % (image_id, i))
            response = requests.put(path, headers=self._headers())
            self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images/%s/tags/fail_me' % image_id)
        response = requests.put(path, headers=self._headers())
        self.assertEqual(http.REQUEST_ENTITY_TOO_LARGE, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        tags = jsonutils.loads(response.text)['tags']
        self.assertEqual(10, len(tags))
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type})
        doc = [{'op': 'replace', 'path': '/tags', 'value': ['foo']}]
        data = jsonutils.dumps(doc)
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.OK, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type})
        tags = ['foo%d' % i for i in range(11)]
        doc = [{'op': 'replace', 'path': '/tags', 'value': tags}]
        data = jsonutils.dumps(doc)
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.REQUEST_ENTITY_TOO_LARGE, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        tags = jsonutils.loads(response.text)['tags']
        self.assertEqual(['foo'], tags)
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type})
        doc = [{'op': 'replace', 'path': '/tags', 'value': ['sniff', 'snozz', 'snozz']}]
        data = jsonutils.dumps(doc)
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.OK, response.status_code)
        tags = jsonutils.loads(response.text)['tags']
        self.assertEqual(['sniff', 'snozz'], sorted(tags))
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        tags = jsonutils.loads(response.text)['tags']
        self.assertEqual(['sniff', 'snozz'], sorted(tags))
        path = self._url('/v2/images/%s/tags/snozz' % image_id)
        response = requests.put(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images/%s/tags/gabe%%40example.com' % image_id)
        response = requests.put(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        tags = jsonutils.loads(response.text)['tags']
        self.assertEqual(['gabe@example.com', 'sniff', 'snozz'], sorted(tags))
        path = self._url('/v2/images?tag=sniff')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(1, len(images))
        self.assertEqual('image-1', images[0]['name'])
        path = self._url('/v2/images?tag=sniff&tag=snozz')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(1, len(images))
        self.assertEqual('image-1', images[0]['name'])
        path = self._url('/v2/images?tag=sniff&status=queued')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(1, len(images))
        self.assertEqual('image-1', images[0]['name'])
        path = self._url('/v2/images?tag=sniff&tag=fake')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        path = self._url('/v2/images/%s/tags/gabe%%40example.com' % image_id)
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        tags = jsonutils.loads(response.text)['tags']
        self.assertEqual(['sniff', 'snozz'], sorted(tags))
        path = self._url('/v2/images/%s/tags/gabe%%40example.com' % image_id)
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.NOT_FOUND, response.status_code)
        path = self._url('/v2/images?tag=gabe%%40example.com')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        big_tag = 'a' * 300
        path = self._url('/v2/images/%s/tags/%s' % (image_id, big_tag))
        response = requests.put(path, headers=self._headers())
        self.assertEqual(http.BAD_REQUEST, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        tags = jsonutils.loads(response.text)['tags']
        self.assertEqual(['sniff', 'snozz'], sorted(tags))
        self.stop_servers()

    def test_images_container(self):
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        first = jsonutils.loads(response.text)['first']
        self.assertEqual(0, len(images))
        self.assertNotIn('next', jsonutils.loads(response.text))
        self.assertEqual('/v2/images', first)
        images = []
        fixtures = [{'name': 'image-3', 'type': 'kernel', 'ping': 'pong', 'container_format': 'ami', 'disk_format': 'ami'}, {'name': 'image-4', 'type': 'kernel', 'ping': 'pong', 'container_format': 'bare', 'disk_format': 'ami'}, {'name': 'image-1', 'type': 'kernel', 'ping': 'pong'}, {'name': 'image-3', 'type': 'ramdisk', 'ping': 'pong'}, {'name': 'image-2', 'type': 'kernel', 'ping': 'ding'}, {'name': 'image-3', 'type': 'kernel', 'ping': 'pong'}, {'name': 'image-2,image-5', 'type': 'kernel', 'ping': 'pong'}]
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        for fixture in fixtures:
            data = jsonutils.dumps(fixture)
            response = requests.post(path, headers=headers, data=data)
            self.assertEqual(http.CREATED, response.status_code)
            images.append(jsonutils.loads(response.text))
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        body = jsonutils.loads(response.text)
        self.assertEqual(7, len(body['images']))
        self.assertEqual('/v2/images', body['first'])
        self.assertNotIn('next', jsonutils.loads(response.text))
        url_template = '/v2/images?created_at=lt:%s'
        path = self._url(url_template % images[0]['created_at'])
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        body = jsonutils.loads(response.text)
        self.assertEqual(0, len(body['images']))
        self.assertEqual(url_template % images[0]['created_at'], urllib.parse.unquote(body['first']))
        url_template = '/v2/images?updated_at=lt:%s'
        path = self._url(url_template % images[2]['updated_at'])
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        body = jsonutils.loads(response.text)
        self.assertGreaterEqual(3, len(body['images']))
        self.assertEqual(url_template % images[2]['updated_at'], urllib.parse.unquote(body['first']))
        url_template = '/v2/images?%s=lt:invalid_value'
        for filter in ['updated_at', 'created_at']:
            path = self._url(url_template % filter)
            response = requests.get(path, headers=self._headers())
            self.assertEqual(http.BAD_REQUEST, response.status_code)
        url_template = '/v2/images?%s=invalid_operator:2015-11-19T12:24:02Z'
        for filter in ['updated_at', 'created_at']:
            path = self._url(url_template % filter)
            response = requests.get(path, headers=self._headers())
            self.assertEqual(http.BAD_REQUEST, response.status_code)
        path = self._url('/v2/images?name=%FF')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.BAD_REQUEST, response.status_code)
        url_template = '/v2/images?name=in:%s'
        filter_value = 'image-1,image-2'
        path = self._url(url_template % filter_value)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        body = jsonutils.loads(response.text)
        self.assertGreaterEqual(3, len(body['images']))
        url_template = '/v2/images?container_format=in:%s'
        filter_value = 'bare,ami'
        path = self._url(url_template % filter_value)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        body = jsonutils.loads(response.text)
        self.assertGreaterEqual(2, len(body['images']))
        url_template = '/v2/images?disk_format=in:%s'
        filter_value = 'bare,ami,iso'
        path = self._url(url_template % filter_value)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        body = jsonutils.loads(response.text)
        self.assertGreaterEqual(2, len(body['images']))
        template_url = '/v2/images?limit=2&sort_dir=asc&sort_key=name&marker=%s&type=kernel&ping=pong'
        path = self._url(template_url % images[2]['id'])
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        body = jsonutils.loads(response.text)
        self.assertEqual(2, len(body['images']))
        response_ids = [image['id'] for image in body['images']]
        self.assertEqual([images[6]['id'], images[0]['id']], response_ids)
        path = self._url(body['next'])
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        body = jsonutils.loads(response.text)
        self.assertEqual(2, len(body['images']))
        response_ids = [image['id'] for image in body['images']]
        self.assertEqual([images[5]['id'], images[1]['id']], response_ids)
        path = self._url(body['next'])
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        body = jsonutils.loads(response.text)
        self.assertEqual(0, len(body['images']))
        path = self._url('/v2/images/%s' % images[0]['id'])
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images?marker=%s' % images[0]['id'])
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.BAD_REQUEST, response.status_code)
        self.stop_servers()

    def test_image_visibility_to_different_users(self):
        self.cleanup()
        self.api_server.deployment_flavor = 'fakeauth'
        kwargs = self.__dict__.copy()
        self.start_servers(**kwargs)
        owners = ['admin', 'tenant1', 'tenant2', 'none']
        visibilities = ['public', 'private', 'shared', 'community']
        for owner in owners:
            for visibility in visibilities:
                path = self._url('/v2/images')
                headers = self._headers({'content-type': 'application/json', 'X-Auth-Token': 'createuser:%s:admin' % owner})
                data = jsonutils.dumps({'name': '%s-%s' % (owner, visibility), 'visibility': visibility})
                response = requests.post(path, headers=headers, data=data)
                self.assertEqual(http.CREATED, response.status_code)

        def list_images(tenant, role='', visibility=None):
            auth_token = 'user:%s:%s' % (tenant, role)
            headers = {'X-Auth-Token': auth_token}
            path = self._url('/v2/images')
            if visibility is not None:
                path += '?visibility=%s' % visibility
            response = requests.get(path, headers=headers)
            self.assertEqual(http.OK, response.status_code)
            return jsonutils.loads(response.text)['images']
        images = list_images('tenant1', role='reader')
        self.assertEqual(7, len(images))
        for image in images:
            self.assertTrue(image['visibility'] == 'public' or 'tenant1' in image['name'])
        images = list_images('tenant1', role='reader', visibility='public')
        self.assertEqual(4, len(images))
        for image in images:
            self.assertEqual('public', image['visibility'])
        images = list_images('tenant1', role='reader', visibility='private')
        self.assertEqual(1, len(images))
        image = images[0]
        self.assertEqual('private', image['visibility'])
        self.assertIn('tenant1', image['name'])
        images = list_images('tenant1', role='reader', visibility='shared')
        self.assertEqual(1, len(images))
        image = images[0]
        self.assertEqual('shared', image['visibility'])
        self.assertIn('tenant1', image['name'])
        images = list_images('tenant1', role='reader', visibility='community')
        self.assertEqual(4, len(images))
        for image in images:
            self.assertEqual('community', image['visibility'])
        images = list_images('none', role='reader')
        self.assertEqual(4, len(images))
        for image in images:
            self.assertEqual('public', image['visibility'])
        images = list_images('none', role='reader', visibility='public')
        self.assertEqual(4, len(images))
        for image in images:
            self.assertEqual('public', image['visibility'])
        images = list_images('none', role='reader', visibility='private')
        self.assertEqual(0, len(images))
        images = list_images('none', role='reader', visibility='shared')
        self.assertEqual(0, len(images))
        images = list_images('none', role='reader', visibility='community')
        self.assertEqual(4, len(images))
        for image in images:
            self.assertEqual('community', image['visibility'])
        images = list_images('none', role='admin')
        self.assertEqual(12, len(images))
        images = list_images('none', role='admin', visibility='public')
        self.assertEqual(4, len(images))
        for image in images:
            self.assertEqual('public', image['visibility'])
        images = list_images('none', role='admin', visibility='private')
        self.assertEqual(4, len(images))
        for image in images:
            self.assertEqual('private', image['visibility'])
        images = list_images('none', role='admin', visibility='shared')
        self.assertEqual(4, len(images))
        for image in images:
            self.assertEqual('shared', image['visibility'])
        images = list_images('none', role='admin', visibility='community')
        self.assertEqual(4, len(images))
        for image in images:
            self.assertEqual('community', image['visibility'])
        images = list_images('admin', role='admin')
        self.assertEqual(13, len(images))
        images = list_images('admin', role='admin', visibility='public')
        self.assertEqual(4, len(images))
        for image in images:
            self.assertEqual('public', image['visibility'])
        images = list_images('admin', role='admin', visibility='private')
        self.assertEqual(4, len(images))
        for image in images:
            self.assertEqual('private', image['visibility'])
        images = list_images('admin', role='admin', visibility='shared')
        self.assertEqual(4, len(images))
        for image in images:
            self.assertEqual('shared', image['visibility'])
        images = list_images('admin', role='admin', visibility='community')
        self.assertEqual(4, len(images))
        for image in images:
            self.assertEqual('community', image['visibility'])
        self.stop_servers()

    def test_update_locations(self):
        self.api_server.show_multiple_locations = True
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        self.assertEqual('queued', image['status'])
        self.assertIsNone(image['size'])
        self.assertIsNone(image['virtual_size'])
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type})
        url = 'http://127.0.0.1:%s/foo_image' % self.http_port0
        data = jsonutils.dumps([{'op': 'replace', 'path': '/locations', 'value': [{'url': url, 'metadata': {}}]}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.OK, response.status_code, response.text)
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=headers)
        self.assertEqual(http.OK, response.status_code)
        image = jsonutils.loads(response.text)
        self.assertEqual(10, image['size'])

    def test_update_locations_with_restricted_sources(self):
        self.api_server.show_multiple_locations = True
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        self.assertEqual('queued', image['status'])
        self.assertIsNone(image['size'])
        self.assertIsNone(image['virtual_size'])
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type})
        data = jsonutils.dumps([{'op': 'replace', 'path': '/locations', 'value': [{'url': 'file:///foo_image', 'metadata': {}}]}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.BAD_REQUEST, response.status_code, response.text)
        data = jsonutils.dumps([{'op': 'replace', 'path': '/locations', 'value': [{'url': 'swift+config:///foo_image', 'metadata': {}}]}])
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.BAD_REQUEST, response.status_code, response.text)