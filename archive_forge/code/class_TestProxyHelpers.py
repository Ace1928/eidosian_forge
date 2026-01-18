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
class TestProxyHelpers(base.IsolatedUnitTest):

    def test_proxy_response_error(self):
        e = glance.api.v2.images.proxy_response_error(123, 'Foo')
        self.assertIsInstance(e, webob.exc.HTTPError)
        self.assertEqual(123, e.code)
        self.assertEqual('123 Foo', e.status)

    def test_is_proxyable(self):
        controller = glance.api.v2.images.ImagesController(None, None, None, None)
        self.config(worker_self_reference_url='http://worker1')
        mock_image = mock.MagicMock(extra_properties={})
        self.assertFalse(controller.is_proxyable(mock_image))
        mock_image.extra_properties['os_glance_stage_host'] = 'http://worker1'
        self.assertFalse(controller.is_proxyable(mock_image))
        mock_image.extra_properties['os_glance_stage_host'] = 'http://worker2'
        self.assertTrue(controller.is_proxyable(mock_image))

    def test_self_url(self):
        controller = glance.api.v2.images.ImagesController(None, None, None, None)
        self.assertIsNone(controller.self_url)
        self.config(public_endpoint='http://lb.example.com')
        self.assertEqual('http://lb.example.com', controller.self_url)
        self.config(worker_self_reference_url='http://worker1.example.com')
        self.assertEqual('http://worker1.example.com', controller.self_url)