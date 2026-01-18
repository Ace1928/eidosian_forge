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
class TestImagesSerializerDirectUrl(test_utils.BaseTestCase):

    def setUp(self):
        super(TestImagesSerializerDirectUrl, self).setUp()
        self.serializer = glance.api.v2.images.ResponseSerializer()
        self.active_image = _domain_fixture(UUID1, name='image-1', visibility='public', status='active', size=1024, virtual_size=3072, created_at=DATETIME, updated_at=DATETIME, locations=[{'id': '1', 'url': 'http://some/fake/location', 'metadata': {}, 'status': 'active'}])
        self.queued_image = _domain_fixture(UUID2, name='image-2', status='active', created_at=DATETIME, updated_at=DATETIME, checksum='ca425b88f047ce8ec45ee90e813ada91')
        self.location_data_image_url = 'http://abc.com/somewhere'
        self.location_data_image_meta = {'key': 98231}
        self.location_data_image = _domain_fixture(UUID2, name='image-2', status='active', created_at=DATETIME, updated_at=DATETIME, locations=[{'id': '2', 'url': self.location_data_image_url, 'metadata': self.location_data_image_meta, 'status': 'active'}])

    def _do_index(self):
        request = webob.Request.blank('/v2/images')
        response = webob.Response(request=request)
        self.serializer.index(response, {'images': [self.active_image, self.queued_image]})
        return jsonutils.loads(response.body)['images']

    def _do_show(self, image):
        request = webob.Request.blank('/v2/images')
        response = webob.Response(request=request)
        self.serializer.show(response, image)
        return jsonutils.loads(response.body)

    def test_index_store_location_enabled(self):
        self.config(show_image_direct_url=True)
        images = self._do_index()
        self.assertEqual(UUID1, images[0]['id'])
        self.assertEqual(UUID2, images[1]['id'])
        self.assertEqual('http://some/fake/location', images[0]['direct_url'])
        self.assertNotIn('direct_url', images[1])

    def test_index_store_multiple_location_enabled(self):
        self.config(show_multiple_locations=True)
        request = webob.Request.blank('/v2/images')
        response = webob.Response(request=request)
        (self.serializer.index(response, {'images': [self.location_data_image]}),)
        images = jsonutils.loads(response.body)['images']
        location = images[0]['locations'][0]
        self.assertEqual(location['url'], self.location_data_image_url)
        self.assertEqual(location['metadata'], self.location_data_image_meta)

    def test_index_store_location_explicitly_disabled(self):
        self.config(show_image_direct_url=False)
        images = self._do_index()
        self.assertNotIn('direct_url', images[0])
        self.assertNotIn('direct_url', images[1])

    def test_show_location_enabled(self):
        self.config(show_image_direct_url=True)
        image = self._do_show(self.active_image)
        self.assertEqual('http://some/fake/location', image['direct_url'])

    def test_show_location_enabled_but_not_set(self):
        self.config(show_image_direct_url=True)
        image = self._do_show(self.queued_image)
        self.assertNotIn('direct_url', image)

    def test_show_location_explicitly_disabled(self):
        self.config(show_image_direct_url=False)
        image = self._do_show(self.active_image)
        self.assertNotIn('direct_url', image)