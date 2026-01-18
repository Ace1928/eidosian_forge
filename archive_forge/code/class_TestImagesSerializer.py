import datetime
import hashlib
import http.client as http
import os
import requests
from unittest import mock
import uuid
from castellan.common import exception as castellan_exception
import glance_store as store
from oslo_config import cfg
from oslo_serialization import jsonutils
from oslo_utils import fixture
import testtools
import webob
import webob.exc
import glance.api.v2.image_actions
import glance.api.v2.images
from glance.common import exception
from glance.common import store_utils
from glance.common import timeutils
from glance import domain
import glance.notifier
import glance.schema
from glance.tests.unit import base
from glance.tests.unit.keymgr import fake as fake_keymgr
import glance.tests.unit.utils as unit_test_utils
from glance.tests.unit.v2 import test_tasks_resource
import glance.tests.utils as test_utils
class TestImagesSerializer(test_utils.BaseTestCase):

    def setUp(self):
        super(TestImagesSerializer, self).setUp()
        self.serializer = glance.api.v2.images.ResponseSerializer()
        self.fixtures = [_domain_fixture(UUID1, name='image-1', size=1024, virtual_size=3072, created_at=DATETIME, updated_at=DATETIME, owner=TENANT1, visibility='public', container_format='ami', tags=['one', 'two'], disk_format='ami', min_ram=128, min_disk=10, checksum='ca425b88f047ce8ec45ee90e813ada91', os_hash_algo=FAKEHASHALGO, os_hash_value=MULTIHASH1), _domain_fixture(UUID2, created_at=DATETIME, updated_at=DATETIME)]

    def test_index(self):
        expected = {'images': [{'id': UUID1, 'name': 'image-1', 'status': 'queued', 'visibility': 'public', 'protected': False, 'os_hidden': False, 'tags': set(['one', 'two']), 'size': 1024, 'virtual_size': 3072, 'checksum': 'ca425b88f047ce8ec45ee90e813ada91', 'os_hash_algo': FAKEHASHALGO, 'os_hash_value': MULTIHASH1, 'container_format': 'ami', 'disk_format': 'ami', 'min_ram': 128, 'min_disk': 10, 'created_at': ISOTIME, 'updated_at': ISOTIME, 'self': '/v2/images/%s' % UUID1, 'file': '/v2/images/%s/file' % UUID1, 'schema': '/v2/schemas/image', 'owner': '6838eb7b-6ded-434a-882c-b344c77fe8df'}, {'id': UUID2, 'status': 'queued', 'visibility': 'private', 'protected': False, 'os_hidden': False, 'tags': set([]), 'created_at': ISOTIME, 'updated_at': ISOTIME, 'self': '/v2/images/%s' % UUID2, 'file': '/v2/images/%s/file' % UUID2, 'schema': '/v2/schemas/image', 'size': None, 'name': None, 'owner': None, 'min_ram': None, 'min_disk': None, 'checksum': None, 'os_hash_algo': None, 'os_hash_value': None, 'disk_format': None, 'virtual_size': None, 'container_format': None}], 'first': '/v2/images', 'schema': '/v2/schemas/images'}
        request = webob.Request.blank('/v2/images')
        response = webob.Response(request=request)
        result = {'images': self.fixtures}
        self.serializer.index(response, result)
        actual = jsonutils.loads(response.body)
        for image in actual['images']:
            image['tags'] = set(image['tags'])
        self.assertEqual(expected, actual)
        self.assertEqual('application/json', response.content_type)

    def test_index_next_marker(self):
        request = webob.Request.blank('/v2/images')
        response = webob.Response(request=request)
        result = {'images': self.fixtures, 'next_marker': UUID2}
        self.serializer.index(response, result)
        output = jsonutils.loads(response.body)
        self.assertEqual('/v2/images?marker=%s' % UUID2, output['next'])

    def test_index_carries_query_parameters(self):
        url = '/v2/images?limit=10&sort_key=id&sort_dir=asc'
        request = webob.Request.blank(url)
        response = webob.Response(request=request)
        result = {'images': self.fixtures, 'next_marker': UUID2}
        self.serializer.index(response, result)
        output = jsonutils.loads(response.body)
        expected_url = '/v2/images?limit=10&sort_dir=asc&sort_key=id'
        self.assertEqual(unit_test_utils.sort_url_by_qs_keys(expected_url), unit_test_utils.sort_url_by_qs_keys(output['first']))
        expect_next = '/v2/images?limit=10&marker=%s&sort_dir=asc&sort_key=id'
        self.assertEqual(unit_test_utils.sort_url_by_qs_keys(expect_next % UUID2), unit_test_utils.sort_url_by_qs_keys(output['next']))

    def test_index_forbidden_get_image_location(self):
        """Make sure the serializer works fine.

        No matter if current user is authorized to get image location if the
        show_multiple_locations is False.

        """

        class ImageLocations(object):

            def __len__(self):
                raise exception.Forbidden()
        self.config(show_multiple_locations=False)
        self.config(show_image_direct_url=False)
        url = '/v2/images?limit=10&sort_key=id&sort_dir=asc'
        request = webob.Request.blank(url)
        response = webob.Response(request=request)
        result = {'images': self.fixtures}
        self.assertEqual(http.OK, response.status_int)
        result['images'][0].locations = ImageLocations()
        self.serializer.index(response, result)
        self.assertEqual(http.OK, response.status_int)

    def test_show_full_fixture(self):
        expected = {'id': UUID1, 'name': 'image-1', 'status': 'queued', 'visibility': 'public', 'protected': False, 'os_hidden': False, 'tags': set(['one', 'two']), 'size': 1024, 'virtual_size': 3072, 'checksum': 'ca425b88f047ce8ec45ee90e813ada91', 'os_hash_algo': FAKEHASHALGO, 'os_hash_value': MULTIHASH1, 'container_format': 'ami', 'disk_format': 'ami', 'min_ram': 128, 'min_disk': 10, 'created_at': ISOTIME, 'updated_at': ISOTIME, 'self': '/v2/images/%s' % UUID1, 'file': '/v2/images/%s/file' % UUID1, 'schema': '/v2/schemas/image', 'owner': '6838eb7b-6ded-434a-882c-b344c77fe8df'}
        response = webob.Response()
        self.serializer.show(response, self.fixtures[0])
        actual = jsonutils.loads(response.body)
        actual['tags'] = set(actual['tags'])
        self.assertEqual(expected, actual)
        self.assertEqual('application/json', response.content_type)

    def test_show_minimal_fixture(self):
        expected = {'id': UUID2, 'status': 'queued', 'visibility': 'private', 'protected': False, 'os_hidden': False, 'tags': [], 'created_at': ISOTIME, 'updated_at': ISOTIME, 'self': '/v2/images/%s' % UUID2, 'file': '/v2/images/%s/file' % UUID2, 'schema': '/v2/schemas/image', 'size': None, 'name': None, 'owner': None, 'min_ram': None, 'min_disk': None, 'checksum': None, 'os_hash_algo': None, 'os_hash_value': None, 'disk_format': None, 'virtual_size': None, 'container_format': None}
        response = webob.Response()
        self.serializer.show(response, self.fixtures[1])
        self.assertEqual(expected, jsonutils.loads(response.body))

    def test_create(self):
        expected = {'id': UUID1, 'name': 'image-1', 'status': 'queued', 'visibility': 'public', 'protected': False, 'os_hidden': False, 'tags': ['one', 'two'], 'size': 1024, 'virtual_size': 3072, 'checksum': 'ca425b88f047ce8ec45ee90e813ada91', 'os_hash_algo': FAKEHASHALGO, 'os_hash_value': MULTIHASH1, 'container_format': 'ami', 'disk_format': 'ami', 'min_ram': 128, 'min_disk': 10, 'created_at': ISOTIME, 'updated_at': ISOTIME, 'self': '/v2/images/%s' % UUID1, 'file': '/v2/images/%s/file' % UUID1, 'schema': '/v2/schemas/image', 'owner': '6838eb7b-6ded-434a-882c-b344c77fe8df'}
        response = webob.Response()
        self.serializer.create(response, self.fixtures[0])
        self.assertEqual(http.CREATED, response.status_int)
        actual = jsonutils.loads(response.body)
        actual['tags'] = sorted(actual['tags'])
        self.assertEqual(expected, actual)
        self.assertEqual('application/json', response.content_type)
        self.assertEqual('/v2/images/%s' % UUID1, response.location)

    def test_create_has_import_methods_header(self):
        header_name = 'OpenStack-image-import-methods'
        enabled_methods = ['one', 'two', 'three']
        self.config(enabled_import_methods=enabled_methods)
        response = webob.Response()
        self.serializer.create(response, self.fixtures[0])
        self.assertEqual(http.CREATED, response.status_int)
        header_value = response.headers.get(header_name)
        self.assertIsNotNone(header_value)
        self.assertCountEqual(enabled_methods, header_value.split(','))
        self.config(enabled_import_methods=['swift-party-time'])
        response = webob.Response()
        self.serializer.create(response, self.fixtures[0])
        self.assertEqual(http.CREATED, response.status_int)
        header_value = response.headers.get(header_name)
        self.assertIsNotNone(header_value)
        self.assertEqual('swift-party-time', header_value)
        self.config(enabled_import_methods=[])
        response = webob.Response()
        self.serializer.create(response, self.fixtures[0])
        self.assertEqual(http.CREATED, response.status_int)
        headers = response.headers.keys()
        self.assertNotIn(header_name, headers)

    def test_update(self):
        expected = {'id': UUID1, 'name': 'image-1', 'status': 'queued', 'visibility': 'public', 'protected': False, 'os_hidden': False, 'tags': set(['one', 'two']), 'size': 1024, 'virtual_size': 3072, 'checksum': 'ca425b88f047ce8ec45ee90e813ada91', 'os_hash_algo': FAKEHASHALGO, 'os_hash_value': MULTIHASH1, 'container_format': 'ami', 'disk_format': 'ami', 'min_ram': 128, 'min_disk': 10, 'created_at': ISOTIME, 'updated_at': ISOTIME, 'self': '/v2/images/%s' % UUID1, 'file': '/v2/images/%s/file' % UUID1, 'schema': '/v2/schemas/image', 'owner': '6838eb7b-6ded-434a-882c-b344c77fe8df'}
        response = webob.Response()
        self.serializer.update(response, self.fixtures[0])
        actual = jsonutils.loads(response.body)
        actual['tags'] = set(actual['tags'])
        self.assertEqual(expected, actual)
        self.assertEqual('application/json', response.content_type)

    def test_import_image(self):
        response = webob.Response()
        self.serializer.import_image(response, {})
        self.assertEqual(http.ACCEPTED, response.status_int)
        self.assertEqual('0', response.headers['Content-Length'])

    def test_image_stage_host_hidden(self):
        response = webob.Response()
        self.serializer.show(response, mock.MagicMock(extra_properties={'foo': 'bar', 'os_glance_stage_host': 'http://foo'}))
        actual = jsonutils.loads(response.body)
        self.assertIn('foo', actual)
        self.assertNotIn('os_glance_stage_host', actual)