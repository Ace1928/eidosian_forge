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
class TestImageDirectURLVisibility(functional.FunctionalTest):

    def setUp(self):
        super(TestImageDirectURLVisibility, self).setUp()
        self.cleanup()
        self.include_scrubber = False
        self.api_server.deployment_flavor = 'noauth'

    def _headers(self, custom_headers=None):
        base_headers = {'X-Identity-Status': 'Confirmed', 'X-Auth-Token': '932c5c84-02ac-4fe5-a9ba-620af0e2bb96', 'X-User-Id': 'f9a41d13-0c13-47e9-bee2-ce4e8bfe958e', 'X-Tenant-Id': TENANT1, 'X-Roles': 'reader,member'}
        base_headers.update(custom_headers or {})
        return base_headers

    def test_image_direct_url_visible(self):
        self.api_server.show_image_direct_url = True
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        data = jsonutils.dumps({'name': 'image-1', 'type': 'kernel', 'foo': 'bar', 'disk_format': 'aki', 'container_format': 'aki', 'visibility': 'public'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        path = self._url('/v2/images/%s' % image_id)
        headers = self._headers({'Content-Type': 'application/json'})
        response = requests.get(path, headers=headers)
        self.assertEqual(http.OK, response.status_code)
        image = jsonutils.loads(response.text)
        self.assertNotIn('direct_url', image)
        path = self._url('/v2/images/%s/file' % image_id)
        headers = self._headers({'Content-Type': 'application/octet-stream'})
        response = requests.put(path, headers=headers, data='ZZZZZ')
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        headers = self._headers({'Content-Type': 'application/json'})
        response = requests.get(path, headers=headers)
        self.assertEqual(http.OK, response.status_code)
        image = jsonutils.loads(response.text)
        self.assertIn('direct_url', image)
        path = self._url('/v2/images/%s' % image_id)
        headers = self._headers({'Content-Type': 'application/json', 'X-Tenant-Id': TENANT2})
        response = requests.get(path, headers=headers)
        self.assertEqual(http.OK, response.status_code)
        image = jsonutils.loads(response.text)
        self.assertIn('direct_url', image)
        path = self._url('/v2/images')
        headers = self._headers({'Content-Type': 'application/json'})
        response = requests.get(path, headers=headers)
        self.assertEqual(http.OK, response.status_code)
        image = jsonutils.loads(response.text)['images'][0]
        self.assertIn('direct_url', image)
        self.stop_servers()

    def test_image_multiple_location_url_visible(self):
        self.api_server.show_multiple_locations = True
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = jsonutils.dumps({'name': 'image-1', 'type': 'kernel', 'foo': 'bar', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        path = self._url('/v2/images/%s' % image_id)
        headers = self._headers({'Content-Type': 'application/json'})
        response = requests.get(path, headers=headers)
        self.assertEqual(http.OK, response.status_code)
        image = jsonutils.loads(response.text)
        self.assertIn('locations', image)
        self.assertEqual([], image['locations'])
        path = self._url('/v2/images/%s/file' % image_id)
        headers = self._headers({'Content-Type': 'application/octet-stream'})
        response = requests.put(path, headers=headers, data='ZZZZZ')
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        headers = self._headers({'Content-Type': 'application/json'})
        response = requests.get(path, headers=headers)
        self.assertEqual(http.OK, response.status_code)
        image = jsonutils.loads(response.text)
        self.assertIn('locations', image)
        loc = image['locations']
        self.assertGreater(len(loc), 0)
        loc = loc[0]
        self.assertIn('url', loc)
        self.assertIn('metadata', loc)
        self.stop_servers()

    def test_image_direct_url_not_visible(self):
        self.api_server.show_image_direct_url = False
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = jsonutils.dumps({'name': 'image-1', 'type': 'kernel', 'foo': 'bar', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
        path = self._url('/v2/images/%s/file' % image_id)
        headers = self._headers({'Content-Type': 'application/octet-stream'})
        response = requests.put(path, headers=headers, data='ZZZZZ')
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        headers = self._headers({'Content-Type': 'application/json'})
        response = requests.get(path, headers=headers)
        self.assertEqual(http.OK, response.status_code)
        image = jsonutils.loads(response.text)
        self.assertNotIn('direct_url', image)
        path = self._url('/v2/images')
        headers = self._headers({'Content-Type': 'application/json'})
        response = requests.get(path, headers=headers)
        self.assertEqual(http.OK, response.status_code)
        image = jsonutils.loads(response.text)['images'][0]
        self.assertNotIn('direct_url', image)
        self.stop_servers()