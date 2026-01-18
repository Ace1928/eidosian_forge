import collections
import http.client as http
import io
from unittest import mock
import copy
import os
import sys
import uuid
import fixtures
from oslo_serialization import jsonutils
import webob
from glance.cmd import replicator as glance_replicator
from glance.common import exception
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils as test_utils
class ImageServiceTestCase(test_utils.BaseTestCase):

    def test_rest_errors(self):
        c = glance_replicator.ImageService(FakeHTTPConnection(), 'noauth')
        for code, exc in [(http.BAD_REQUEST, webob.exc.HTTPBadRequest), (http.UNAUTHORIZED, webob.exc.HTTPUnauthorized), (http.FORBIDDEN, webob.exc.HTTPForbidden), (http.CONFLICT, webob.exc.HTTPConflict), (http.INTERNAL_SERVER_ERROR, webob.exc.HTTPInternalServerError)]:
            c.conn.prime_request('GET', 'v1/images/5dcddce0-cba5-4f18-9cf4-9853c7b207a6', '', {'x-auth-token': 'noauth'}, code, '', {})
            self.assertRaises(exc, c.get_image, '5dcddce0-cba5-4f18-9cf4-9853c7b207a6')

    def test_rest_get_images(self):
        c = glance_replicator.ImageService(FakeHTTPConnection(), 'noauth')
        resp = {'images': [IMG_RESPONSE_ACTIVE, IMG_RESPONSE_QUEUED]}
        c.conn.prime_request('GET', 'v1/images/detail?is_public=None', '', {'x-auth-token': 'noauth'}, http.OK, jsonutils.dumps(resp), {})
        c.conn.prime_request('GET', 'v1/images/detail?marker=%s&is_public=None' % IMG_RESPONSE_QUEUED['id'], '', {'x-auth-token': 'noauth'}, http.OK, jsonutils.dumps({'images': []}), {})
        imgs = list(c.get_images())
        self.assertEqual(2, len(imgs))
        self.assertEqual(2, c.conn.count)

    def test_rest_get_image(self):
        c = glance_replicator.ImageService(FakeHTTPConnection(), 'noauth')
        image_contents = 'THISISTHEIMAGEBODY'
        c.conn.prime_request('GET', 'v1/images/%s' % IMG_RESPONSE_ACTIVE['id'], '', {'x-auth-token': 'noauth'}, http.OK, image_contents, IMG_RESPONSE_ACTIVE)
        body = c.get_image(IMG_RESPONSE_ACTIVE['id'])
        self.assertEqual(image_contents, body.read())

    def test_rest_header_list_to_dict(self):
        i = [('x-image-meta-banana', 42), ('gerkin', 12), ('x-image-meta-property-frog', 11), ('x-image-meta-property-duck', 12)]
        o = glance_replicator.ImageService._header_list_to_dict(i)
        self.assertIn('banana', o)
        self.assertIn('gerkin', o)
        self.assertIn('properties', o)
        self.assertIn('frog', o['properties'])
        self.assertIn('duck', o['properties'])
        self.assertNotIn('x-image-meta-banana', o)

    def test_rest_get_image_meta(self):
        c = glance_replicator.ImageService(FakeHTTPConnection(), 'noauth')
        c.conn.prime_request('HEAD', 'v1/images/%s' % IMG_RESPONSE_ACTIVE['id'], '', {'x-auth-token': 'noauth'}, http.OK, '', IMG_RESPONSE_ACTIVE)
        header = c.get_image_meta(IMG_RESPONSE_ACTIVE['id'])
        self.assertIn('id', header)

    def test_rest_dict_to_headers(self):
        i = {'banana': 42, 'gerkin': 12, 'properties': {'frog': 1, 'kernel_id': None}}
        o = glance_replicator.ImageService._dict_to_headers(i)
        self.assertIn('x-image-meta-banana', o)
        self.assertIn('x-image-meta-gerkin', o)
        self.assertIn('x-image-meta-property-frog', o)
        self.assertIn('x-image-meta-property-kernel_id', o)
        self.assertEqual(o['x-image-meta-property-kernel_id'], '')
        self.assertNotIn('properties', o)

    def test_rest_add_image(self):
        c = glance_replicator.ImageService(FakeHTTPConnection(), 'noauth')
        image_body = 'THISISANIMAGEBODYFORSURE!'
        image_meta_with_proto = {'x-auth-token': 'noauth', 'Content-Type': 'application/octet-stream', 'Content-Length': len(image_body)}
        for key in IMG_RESPONSE_ACTIVE:
            image_meta_with_proto['x-image-meta-%s' % key] = IMG_RESPONSE_ACTIVE[key]
        c.conn.prime_request('POST', 'v1/images', image_body, image_meta_with_proto, http.OK, '', IMG_RESPONSE_ACTIVE)
        headers, body = c.add_image(IMG_RESPONSE_ACTIVE, image_body)
        self.assertEqual(IMG_RESPONSE_ACTIVE, headers)
        self.assertEqual(1, c.conn.count)

    def test_rest_add_image_meta(self):
        c = glance_replicator.ImageService(FakeHTTPConnection(), 'noauth')
        image_meta = {'id': '5dcddce0-cba5-4f18-9cf4-9853c7b207a6'}
        image_meta_headers = glance_replicator.ImageService._dict_to_headers(image_meta)
        image_meta_headers['x-auth-token'] = 'noauth'
        image_meta_headers['Content-Type'] = 'application/octet-stream'
        c.conn.prime_request('PUT', 'v1/images/%s' % image_meta['id'], '', image_meta_headers, http.OK, '', '')
        headers, body = c.add_image_meta(image_meta)