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
class TestImageMembers(functional.FunctionalTest):

    def setUp(self):
        super(TestImageMembers, self).setUp()
        self.cleanup()
        self.include_scrubber = False
        self.api_server.deployment_flavor = 'fakeauth'
        self.start_servers(**self.__dict__.copy())

    def _headers(self, custom_headers=None):
        base_headers = {'X-Identity-Status': 'Confirmed', 'X-Auth-Token': '932c5c84-02ac-4fe5-a9ba-620af0e2bb96', 'X-User-Id': 'f9a41d13-0c13-47e9-bee2-ce4e8bfe958e', 'X-Tenant-Id': TENANT1, 'X-Roles': 'reader,member'}
        base_headers.update(custom_headers or {})
        return base_headers

    def test_image_member_lifecycle(self):
        path = self._url('/v2/images')
        response = requests.get(path, headers=get_auth_header('tenant1'))
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        owners = ['tenant1', 'tenant2', 'admin']
        visibilities = ['community', 'private', 'public', 'shared']
        image_fixture = []
        for owner in owners:
            for visibility in visibilities:
                path = self._url('/v2/images')
                role = 'member'
                if visibility == 'public':
                    role = 'admin'
                headers = self._headers({'content-type': 'application/json', 'X-Auth-Token': 'createuser:%s:admin' % owner, 'X-Roles': role})
                data = jsonutils.dumps({'name': '%s-%s' % (owner, visibility), 'visibility': visibility})
                response = requests.post(path, headers=headers, data=data)
                self.assertEqual(http.CREATED, response.status_code)
                image_fixture.append(jsonutils.loads(response.text))
        path = self._url('/v2/images')
        response = requests.get(path, headers=get_auth_header('tenant1'))
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(6, len(images))
        path = self._url('/v2/images')
        response = requests.get(path, headers=get_auth_header(TENANT3))
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(3, len(images))
        path = self._url('/v2/images/%s/members' % image_fixture[3]['id'])
        body = jsonutils.dumps({'member': TENANT3})
        response = requests.post(path, headers=get_auth_header('tenant1'), data=body)
        self.assertEqual(http.OK, response.status_code)
        image_member = jsonutils.loads(response.text)
        self.assertEqual(image_fixture[3]['id'], image_member['image_id'])
        self.assertEqual(TENANT3, image_member['member_id'])
        self.assertIn('created_at', image_member)
        self.assertIn('updated_at', image_member)
        self.assertEqual('pending', image_member['status'])
        path = self._url('/v2/images')
        response = requests.get(path, headers=get_auth_header(TENANT3))
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(3, len(images))
        path = self._url('/v2/images?visibility=shared')
        response = requests.get(path, headers=get_auth_header(TENANT3))
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        path = self._url('/v2/images?member_status=pending')
        response = requests.get(path, headers=get_auth_header(TENANT3))
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(4, len(images))
        path = self._url('/v2/images?member_status=all')
        response = requests.get(path, headers=get_auth_header(TENANT3))
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(4, len(images))
        path = self._url('/v2/images?member_status=pending&visibility=shared')
        response = requests.get(path, headers=get_auth_header(TENANT3))
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(1, len(images))
        self.assertEqual(images[0]['name'], 'tenant1-shared')
        path = self._url('/v2/images?member_status=rejected&visibility=shared')
        response = requests.get(path, headers=get_auth_header(TENANT3))
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        path = self._url('/v2/images?member_status=accepted&visibility=shared')
        response = requests.get(path, headers=get_auth_header(TENANT3))
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        path = self._url('/v2/images?visibility=private')
        response = requests.get(path, headers=get_auth_header(TENANT3))
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        path = self._url('/v2/images/%s/members' % image_fixture[7]['id'])
        response = requests.get(path, headers=get_auth_header('tenant2'))
        self.assertEqual(http.OK, response.status_code)
        body = jsonutils.loads(response.text)
        self.assertEqual(0, len(body['members']))
        path = self._url('/v2/images/%s/members/%s' % (image_fixture[3]['id'], TENANT3))
        body = jsonutils.dumps({'status': 'accepted'})
        response = requests.put(path, headers=get_auth_header('tenant1'), data=body)
        self.assertEqual(http.FORBIDDEN, response.status_code)
        path = self._url('/v2/images/%s/members/%s' % (image_fixture[3]['id'], TENANT3))
        response = requests.get(path, headers=get_auth_header('tenant1'))
        self.assertEqual(http.OK, response.status_code)
        body = jsonutils.loads(response.text)
        self.assertEqual('pending', body['status'])
        self.assertEqual(image_fixture[3]['id'], body['image_id'])
        self.assertEqual(TENANT3, body['member_id'])
        path = self._url('/v2/images/%s/members/%s' % (image_fixture[3]['id'], TENANT3))
        response = requests.get(path, headers=get_auth_header(TENANT3))
        self.assertEqual(http.OK, response.status_code)
        body = jsonutils.loads(response.text)
        self.assertEqual('pending', body['status'])
        self.assertEqual(image_fixture[3]['id'], body['image_id'])
        self.assertEqual(TENANT3, body['member_id'])
        path = self._url('/v2/images/%s/members/%s' % (image_fixture[3]['id'], TENANT3))
        response = requests.get(path, headers=get_auth_header('tenant2'))
        self.assertEqual(http.NOT_FOUND, response.status_code)
        path = self._url('/v2/images/%s/members/%s' % (image_fixture[3]['id'], TENANT3))
        body = jsonutils.dumps({'status': 'accepted'})
        response = requests.put(path, headers=get_auth_header(TENANT3), data=body)
        self.assertEqual(http.OK, response.status_code)
        image_member = jsonutils.loads(response.text)
        self.assertEqual(image_fixture[3]['id'], image_member['image_id'])
        self.assertEqual(TENANT3, image_member['member_id'])
        self.assertEqual('accepted', image_member['status'])
        path = self._url('/v2/images')
        response = requests.get(path, headers=get_auth_header(TENANT3))
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(4, len(images))
        path = self._url('/v2/images/%s/members/%s' % (image_fixture[3]['id'], TENANT3))
        body = jsonutils.dumps({'status': 'invalid-status'})
        response = requests.put(path, headers=get_auth_header(TENANT3), data=body)
        self.assertEqual(http.BAD_REQUEST, response.status_code)
        image_id = image_fixture[3]['id']
        path = self._url('/v2/images/%s/stage' % image_id)
        headers = get_auth_header('tenant1')
        headers.update({'Content-Type': 'application/octet-stream'})
        image_data = b'YYYYY'
        response = requests.put(path, headers=headers, data=image_data)
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images/%s/stage' % image_id)
        image_data = b'YYYYY'
        headers.update(get_auth_header(TENANT3))
        response = requests.put(path, headers=headers, data=image_data)
        self.assertEqual(http.FORBIDDEN, response.status_code)
        path = self._url('/v2/images/%s/members/%s' % (image_fixture[3]['id'], TENANT3))
        body = jsonutils.dumps({'status': 'accepted'})
        response = requests.put(path, headers=get_auth_header('tenant1'), data=body)
        self.assertEqual(http.FORBIDDEN, response.status_code)
        path = self._url('/v2/images/%s/members' % image_fixture[7]['id'])
        body = jsonutils.dumps({'member': TENANT4})
        response = requests.post(path, headers=get_auth_header('tenant2'), data=body)
        self.assertEqual(http.OK, response.status_code)
        image_member = jsonutils.loads(response.text)
        self.assertEqual(image_fixture[7]['id'], image_member['image_id'])
        self.assertEqual(TENANT4, image_member['member_id'])
        self.assertIn('created_at', image_member)
        self.assertIn('updated_at', image_member)
        path = self._url('/v2/images/%s/members' % image_fixture[2]['id'])
        body = jsonutils.dumps({'member': TENANT2})
        response = requests.post(path, headers=get_auth_header('tenant1'), data=body)
        self.assertEqual(http.FORBIDDEN, response.status_code)
        path = self._url('/v2/images/%s/members' % image_fixture[1]['id'])
        body = jsonutils.dumps({'member': TENANT2})
        response = requests.post(path, headers=get_auth_header('tenant1'), data=body)
        self.assertEqual(http.FORBIDDEN, response.status_code)
        path = self._url('/v2/images/%s/members' % image_fixture[0]['id'])
        body = jsonutils.dumps({'member': TENANT2})
        response = requests.post(path, headers=get_auth_header('tenant1'), data=body)
        self.assertEqual(http.FORBIDDEN, response.status_code)
        path = self._url('/v2/images/%s/members' % image_fixture[3]['id'])
        response = requests.get(path, headers=get_auth_header('tenant1'))
        self.assertEqual(http.OK, response.status_code)
        body = jsonutils.loads(response.text)
        self.assertEqual(1, len(body['members']))
        path = self._url('/v2/images/%s/members' % image_fixture[3]['id'])
        response = requests.get(path, headers=get_auth_header('tenant1', role='admin'))
        self.assertEqual(http.OK, response.status_code)
        body = jsonutils.loads(response.text)
        self.assertEqual(1, len(body['members']))
        path = self._url('/v2/images/%s/members' % image_fixture[7]['id'])
        response = requests.get(path, headers=get_auth_header('tenant1'))
        self.assertEqual(http.NOT_FOUND, response.status_code)
        path = self._url('/v2/images/%s/members' % image_fixture[2]['id'])
        response = requests.get(path, headers=get_auth_header('tenant1'))
        self.assertIn('Only shared images have members', response.text)
        self.assertEqual(http.FORBIDDEN, response.status_code)
        path = self._url('/v2/images/%s/members' % image_fixture[0]['id'])
        response = requests.get(path, headers=get_auth_header('tenant1'))
        self.assertIn('Only shared images have members', response.text)
        self.assertEqual(http.FORBIDDEN, response.status_code)
        path = self._url('/v2/images/%s/members' % image_fixture[1]['id'])
        response = requests.get(path, headers=get_auth_header('tenant1'))
        self.assertIn('Only shared images have members', response.text)
        self.assertEqual(http.FORBIDDEN, response.status_code)
        path = self._url('/v2/images/%s/members/%s' % (image_fixture[3]['id'], TENANT3))
        response = requests.delete(path, headers=get_auth_header(TENANT3))
        self.assertEqual(http.FORBIDDEN, response.status_code)
        path = self._url('/v2/images/%s/members/%s' % (image_fixture[3]['id'], TENANT3))
        response = requests.delete(path, headers=get_auth_header('tenant1'))
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images/%s/members' % image_fixture[3]['id'])
        response = requests.get(path, headers=get_auth_header('tenant1'))
        self.assertEqual(http.OK, response.status_code)
        body = jsonutils.loads(response.text)
        self.assertEqual(0, len(body['members']))
        path = self._url('/v2/images/%s/members' % image_fixture[3]['id'])
        for i in range(10):
            body = jsonutils.dumps({'member': str(uuid.uuid4())})
            response = requests.post(path, headers=get_auth_header('tenant1'), data=body)
            self.assertEqual(http.OK, response.status_code)
        body = jsonutils.dumps({'member': str(uuid.uuid4())})
        response = requests.post(path, headers=get_auth_header('tenant1'), data=body)
        self.assertEqual(http.REQUEST_ENTITY_TOO_LARGE, response.status_code)
        path = self._url('/v2/images/%s/members/%s' % (image_fixture[2]['id'], TENANT3))
        response = requests.get(path, headers=get_auth_header('tenant1'))
        self.assertEqual(http.NOT_FOUND, response.status_code)
        path = self._url('/v2/images/%s/members/%s' % (image_fixture[0]['id'], TENANT3))
        response = requests.get(path, headers=get_auth_header('tenant1'))
        self.assertEqual(http.NOT_FOUND, response.status_code)
        path = self._url('/v2/images/%s/members/%s' % (image_fixture[1]['id'], TENANT3))
        response = requests.get(path, headers=get_auth_header('tenant1'))
        self.assertEqual(http.NOT_FOUND, response.status_code)
        path = self._url('/v2/images/%s/members/%s' % (image_fixture[2]['id'], TENANT3))
        response = requests.delete(path, headers=get_auth_header('tenant1'))
        self.assertEqual(http.FORBIDDEN, response.status_code)
        path = self._url('/v2/images/%s/members/%s' % (image_fixture[0]['id'], TENANT3))
        response = requests.delete(path, headers=get_auth_header('tenant1'))
        self.assertEqual(http.FORBIDDEN, response.status_code)
        path = self._url('/v2/images/%s/members/%s' % (image_fixture[1]['id'], TENANT3))
        response = requests.delete(path, headers=get_auth_header('tenant1'))
        self.assertEqual(http.FORBIDDEN, response.status_code)
        self.stop_servers()