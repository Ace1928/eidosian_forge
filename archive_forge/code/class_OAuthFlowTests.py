import copy
import datetime
import random
from unittest import mock
import uuid
import freezegun
import http.client
from oslo_serialization import jsonutils
from pycadf import cadftaxonomy
import urllib
from urllib import parse as urlparse
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import oauth1
from keystone.oauth1.backends import base
from keystone.tests import unit
from keystone.tests.unit.common import test_notifications
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
from keystone.tests.unit import test_v3
class OAuthFlowTests(OAuth1Tests):

    def test_oauth_flow(self):
        consumer = self._create_single_consumer()
        consumer_id = consumer['id']
        consumer_secret = consumer['secret']
        self.consumer = {'key': consumer_id, 'secret': consumer_secret}
        self.assertIsNotNone(self.consumer['secret'])
        url, headers = self._create_request_token(self.consumer, self.project_id)
        content = self.post(url, headers=headers, response_content_type='application/x-www-form-urlencoded')
        credentials = _urllib_parse_qs_text_keys(content.result)
        request_key = credentials['oauth_token'][0]
        request_secret = credentials['oauth_token_secret'][0]
        self.request_token = oauth1.Token(request_key, request_secret)
        self.assertIsNotNone(self.request_token.key)
        url = self._authorize_request_token(request_key)
        body = {'roles': [{'id': self.role_id}]}
        resp = self.put(url, body=body, expected_status=http.client.OK)
        self.verifier = resp.result['token']['oauth_verifier']
        self.assertTrue(all((i in base.VERIFIER_CHARS for i in self.verifier)))
        self.assertEqual(8, len(self.verifier))
        self.request_token.set_verifier(self.verifier)
        url, headers = self._create_access_token(self.consumer, self.request_token)
        content = self.post(url, headers=headers, response_content_type='application/x-www-form-urlencoded')
        credentials = _urllib_parse_qs_text_keys(content.result)
        access_key = credentials['oauth_token'][0]
        access_secret = credentials['oauth_token_secret'][0]
        self.access_token = oauth1.Token(access_key, access_secret)
        self.assertIsNotNone(self.access_token.key)
        url, headers, body = self._get_oauth_token(self.consumer, self.access_token)
        content = self.post(url, headers=headers, body=body)
        self.keystone_token_id = content.headers['X-Subject-Token']
        self.keystone_token = content.result['token']
        self.assertIsNotNone(self.keystone_token_id)
        new_role = {'id': uuid.uuid4().hex, 'name': uuid.uuid4().hex}
        PROVIDERS.role_api.create_role(new_role['id'], new_role)
        PROVIDERS.assignment_api.add_role_to_user_and_project(user_id=self.user_id, project_id=self.project_id, role_id=new_role['id'])
        content = self.post(url, headers=headers, body=body)
        token = content.result['token']
        token_roles = [r['id'] for r in token['roles']]
        self.assertIn(self.role_id, token_roles)
        self.assertNotIn(new_role['id'], token_roles)