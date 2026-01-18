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
class MaliciousOAuth1Tests(OAuth1Tests):

    def _switch_baseurl_scheme(self):
        """Switch the base url scheme."""
        base_url_list = list(urlparse.urlparse(self.base_url))
        base_url_list[0] = 'https' if base_url_list[0] == 'http' else 'http'
        bad_url = urlparse.urlunparse(base_url_list)
        return bad_url

    def test_bad_consumer_secret(self):
        consumer = self._create_single_consumer()
        consumer_id = consumer['id']
        consumer = {'key': consumer_id, 'secret': uuid.uuid4().hex}
        url, headers = self._create_request_token(consumer, self.project_id)
        self.post(url, headers=headers, expected_status=http.client.UNAUTHORIZED)

    def test_bad_request_url(self):
        consumer = self._create_single_consumer()
        consumer_id = consumer['id']
        consumer_secret = consumer['secret']
        consumer = {'key': consumer_id, 'secret': consumer_secret}
        bad_base_url = 'http://localhost/identity_admin/v3'
        url, headers = self._create_request_token(consumer, self.project_id, base_url=bad_base_url)
        self.post(url, headers=headers, expected_status=http.client.UNAUTHORIZED)

    def test_bad_request_url_scheme(self):
        consumer = self._create_single_consumer()
        consumer_id = consumer['id']
        consumer_secret = consumer['secret']
        consumer = {'key': consumer_id, 'secret': consumer_secret}
        bad_url_scheme = self._switch_baseurl_scheme()
        url, headers = self._create_request_token(consumer, self.project_id, base_url=bad_url_scheme)
        self.post(url, headers=headers, expected_status=http.client.UNAUTHORIZED)

    def test_bad_request_token_key(self):
        consumer = self._create_single_consumer()
        consumer_id = consumer['id']
        consumer_secret = consumer['secret']
        consumer = {'key': consumer_id, 'secret': consumer_secret}
        url, headers = self._create_request_token(consumer, self.project_id)
        self.post(url, headers=headers, response_content_type='application/x-www-form-urlencoded')
        url = self._authorize_request_token(uuid.uuid4().hex)
        body = {'roles': [{'id': self.role_id}]}
        self.put(url, body=body, expected_status=http.client.NOT_FOUND)

    def test_bad_request_body_when_authorize(self):
        consumer = self._create_single_consumer()
        consumer_id = consumer['id']
        consumer_secret = consumer['secret']
        consumer = {'key': consumer_id, 'secret': consumer_secret}
        url, headers = self._create_request_token(consumer, self.project_id)
        content = self.post(url, headers=headers, response_content_type='application/x-www-form-urlencoded')
        credentials = _urllib_parse_qs_text_keys(content.result)
        request_key = credentials['oauth_token'][0]
        url = self._authorize_request_token(request_key)
        bad_body = {'roles': [{'fake_key': 'fake_value'}]}
        self.put(url, body=bad_body, expected_status=http.client.BAD_REQUEST)

    def test_bad_consumer_id(self):
        consumer = self._create_single_consumer()
        consumer_id = uuid.uuid4().hex
        consumer_secret = consumer['secret']
        consumer = {'key': consumer_id, 'secret': consumer_secret}
        url, headers = self._create_request_token(consumer, self.project_id)
        self.post(url, headers=headers, expected_status=http.client.NOT_FOUND)

    def test_bad_requested_project_id(self):
        consumer = self._create_single_consumer()
        consumer_id = consumer['id']
        consumer_secret = consumer['secret']
        consumer = {'key': consumer_id, 'secret': consumer_secret}
        project_id = uuid.uuid4().hex
        url, headers = self._create_request_token(consumer, project_id)
        self.post(url, headers=headers, expected_status=http.client.NOT_FOUND)

    def test_bad_verifier(self):
        self.config_fixture.config(debug=True, insecure_debug=True)
        consumer = self._create_single_consumer()
        consumer_id = consumer['id']
        consumer_secret = consumer['secret']
        consumer = {'key': consumer_id, 'secret': consumer_secret}
        url, headers = self._create_request_token(consumer, self.project_id)
        content = self.post(url, headers=headers, response_content_type='application/x-www-form-urlencoded')
        credentials = _urllib_parse_qs_text_keys(content.result)
        request_key = credentials['oauth_token'][0]
        request_secret = credentials['oauth_token_secret'][0]
        request_token = oauth1.Token(request_key, request_secret)
        url = self._authorize_request_token(request_key)
        body = {'roles': [{'id': self.role_id}]}
        resp = self.put(url, body=body, expected_status=http.client.OK)
        verifier = resp.result['token']['oauth_verifier']
        self.assertIsNotNone(verifier)
        request_token.set_verifier(uuid.uuid4().hex)
        url, headers = self._create_access_token(consumer, request_token)
        resp = self.post(url, headers=headers, expected_status=http.client.BAD_REQUEST)
        resp_data = jsonutils.loads(resp.body)
        self.assertIn('Validation failed with errors', resp_data.get('error', {}).get('message'))

    def test_validate_access_token_request_failed(self):
        self.config_fixture.config(debug=True, insecure_debug=True)
        consumer = self._create_single_consumer()
        consumer_id = consumer['id']
        consumer_secret = consumer['secret']
        consumer = {'key': consumer_id, 'secret': consumer_secret}
        url, headers = self._create_request_token(consumer, self.project_id)
        content = self.post(url, headers=headers, response_content_type='application/x-www-form-urlencoded')
        credentials = _urllib_parse_qs_text_keys(content.result)
        request_key = credentials['oauth_token'][0]
        request_secret = credentials['oauth_token_secret'][0]
        request_token = oauth1.Token(request_key, request_secret)
        url = self._authorize_request_token(request_key)
        body = {'roles': [{'id': self.role_id}]}
        resp = self.put(url, body=body, expected_status=http.client.OK)
        verifier = resp.result['token']['oauth_verifier']
        request_token.set_verifier(verifier)
        base_url = 'http://localhost/identity_admin/v3'
        url, headers = self._create_access_token(consumer, request_token, base_url=base_url)
        resp = self.post(url, headers=headers, expected_status=http.client.UNAUTHORIZED)
        resp_data = jsonutils.loads(resp.body)
        self.assertIn('Invalid signature', resp_data.get('error', {}).get('message'))
        bad_url_scheme = self._switch_baseurl_scheme()
        url, headers = self._create_access_token(consumer, request_token, base_url=bad_url_scheme)
        resp = self.post(url, headers=headers, expected_status=http.client.UNAUTHORIZED)
        resp_data = jsonutils.loads(resp.body)
        self.assertIn('Invalid signature', resp_data.get('error', {}).get('message'))
        consumer.update({'secret': uuid.uuid4().hex})
        url, headers = self._create_access_token(consumer, request_token)
        resp = self.post(url, headers=headers, expected_status=http.client.UNAUTHORIZED)
        resp_data = jsonutils.loads(resp.body)
        self.assertIn('Invalid signature', resp_data.get('error', {}).get('message'))
        verifier = ''.join(random.SystemRandom().sample(base.VERIFIER_CHARS, 8))
        request_token.set_verifier(verifier)
        url, headers = self._create_access_token(consumer, request_token)
        resp = self.post(url, headers=headers, expected_status=http.client.UNAUTHORIZED)
        resp_data = jsonutils.loads(resp.body)
        self.assertIn('Provided verifier', resp_data.get('error', {}).get('message'))
        consumer.update({'key': uuid.uuid4().hex})
        url, headers = self._create_access_token(consumer, request_token)
        resp = self.post(url, headers=headers, expected_status=http.client.UNAUTHORIZED)
        resp_data = jsonutils.loads(resp.body)
        self.assertIn('Provided consumer does not exist', resp_data.get('error', {}).get('message'))
        consumer2 = self._create_single_consumer()
        consumer.update({'key': consumer2['id']})
        url, headers = self._create_access_token(consumer, request_token)
        resp = self.post(url, headers=headers, expected_status=http.client.UNAUTHORIZED)
        resp_data = jsonutils.loads(resp.body)
        self.assertIn('Provided consumer key', resp_data.get('error', {}).get('message'))

    def test_bad_authorizing_roles_id(self):
        consumer = self._create_single_consumer()
        consumer_id = consumer['id']
        consumer_secret = consumer['secret']
        consumer = {'key': consumer_id, 'secret': consumer_secret}
        new_role = {'id': uuid.uuid4().hex, 'name': uuid.uuid4().hex}
        PROVIDERS.role_api.create_role(new_role['id'], new_role)
        PROVIDERS.assignment_api.add_role_to_user_and_project(user_id=self.user_id, project_id=self.project_id, role_id=new_role['id'])
        url, headers = self._create_request_token(consumer, self.project_id)
        content = self.post(url, headers=headers, response_content_type='application/x-www-form-urlencoded')
        credentials = _urllib_parse_qs_text_keys(content.result)
        request_key = credentials['oauth_token'][0]
        PROVIDERS.assignment_api.remove_role_from_user_and_project(self.user_id, self.project_id, new_role['id'])
        url = self._authorize_request_token(request_key)
        body = {'roles': [{'id': new_role['id']}]}
        self.put(path=url, body=body, expected_status=http.client.UNAUTHORIZED)

    def test_bad_authorizing_roles_name(self):
        consumer = self._create_single_consumer()
        consumer_id = consumer['id']
        consumer_secret = consumer['secret']
        consumer = {'key': consumer_id, 'secret': consumer_secret}
        url, headers = self._create_request_token(consumer, self.project_id)
        content = self.post(url, headers=headers, response_content_type='application/x-www-form-urlencoded')
        credentials = _urllib_parse_qs_text_keys(content.result)
        request_key = credentials['oauth_token'][0]
        url = self._authorize_request_token(request_key)
        body = {'roles': [{'name': 'fake_name'}]}
        self.put(path=url, body=body, expected_status=http.client.NOT_FOUND)

    def test_no_authorizing_user_id(self):
        consumer = self._create_single_consumer()
        consumer_id = consumer['id']
        consumer_secret = consumer['secret']
        consumer = {'key': consumer_id, 'secret': consumer_secret}
        url, headers = self._create_request_token(consumer, self.project_id)
        content = self.post(url, headers=headers, response_content_type='application/x-www-form-urlencoded')
        credentials = _urllib_parse_qs_text_keys(content.result)
        request_key = credentials['oauth_token'][0]
        request_secret = credentials['oauth_token_secret'][0]
        request_token = oauth1.Token(request_key, request_secret)
        url = self._authorize_request_token(request_key)
        body = {'roles': [{'id': self.role_id}]}
        resp = self.put(url, body=body, expected_status=http.client.OK)
        verifier = resp.result['token']['oauth_verifier']
        request_token.set_verifier(verifier)
        request_token_created = PROVIDERS.oauth_api.get_request_token(request_key.decode('utf-8'))
        request_token_created.update({'authorizing_user_id': ''})
        with mock.patch.object(PROVIDERS.oauth_api, 'get_request_token') as mock_token:
            mock_token.return_value = request_token_created
            url, headers = self._create_access_token(consumer, request_token)
            self.post(url, headers=headers, expected_status=http.client.UNAUTHORIZED)

    def test_validate_requet_token_request_failed(self):
        self.config_fixture.config(debug=True, insecure_debug=True)
        consumer = self._create_single_consumer()
        consumer_id = consumer['id']
        consumer_secret = consumer['secret']
        consumer = {'key': consumer_id, 'secret': consumer_secret}
        url = '/OS-OAUTH1/request_token'
        auth_header = 'OAuth oauth_version="1.0", oauth_consumer_key=' + consumer_id
        faked_header = {'Authorization': auth_header, 'requested_project_id': self.project_id}
        resp = self.post(url, headers=faked_header, expected_status=http.client.BAD_REQUEST)
        resp_data = jsonutils.loads(resp.body)
        self.assertIn('Validation failed with errors', resp_data['error']['message'])

    def test_expired_authorizing_request_token(self):
        with freezegun.freeze_time(datetime.datetime.utcnow()) as frozen_time:
            self.config_fixture.config(group='oauth1', request_token_duration=1)
            consumer = self._create_single_consumer()
            consumer_id = consumer['id']
            consumer_secret = consumer['secret']
            self.consumer = {'key': consumer_id, 'secret': consumer_secret}
            self.assertIsNotNone(self.consumer['key'])
            url, headers = self._create_request_token(self.consumer, self.project_id)
            content = self.post(url, headers=headers, response_content_type='application/x-www-form-urlencoded')
            credentials = _urllib_parse_qs_text_keys(content.result)
            request_key = credentials['oauth_token'][0]
            request_secret = credentials['oauth_token_secret'][0]
            self.request_token = oauth1.Token(request_key, request_secret)
            self.assertIsNotNone(self.request_token.key)
            url = self._authorize_request_token(request_key)
            body = {'roles': [{'id': self.role_id}]}
            frozen_time.tick(delta=datetime.timedelta(seconds=CONF.oauth1.request_token_duration + 1))
            self.put(url, body=body, expected_status=http.client.UNAUTHORIZED)

    def test_expired_creating_keystone_token(self):
        with freezegun.freeze_time(datetime.datetime.utcnow()) as frozen_time:
            self.config_fixture.config(group='oauth1', access_token_duration=1)
            consumer = self._create_single_consumer()
            consumer_id = consumer['id']
            consumer_secret = consumer['secret']
            self.consumer = {'key': consumer_id, 'secret': consumer_secret}
            self.assertIsNotNone(self.consumer['key'])
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
            self.request_token.set_verifier(self.verifier)
            url, headers = self._create_access_token(self.consumer, self.request_token)
            content = self.post(url, headers=headers, response_content_type='application/x-www-form-urlencoded')
            credentials = _urllib_parse_qs_text_keys(content.result)
            access_key = credentials['oauth_token'][0]
            access_secret = credentials['oauth_token_secret'][0]
            self.access_token = oauth1.Token(access_key, access_secret)
            self.assertIsNotNone(self.access_token.key)
            url, headers, body = self._get_oauth_token(self.consumer, self.access_token)
            frozen_time.tick(delta=datetime.timedelta(seconds=CONF.oauth1.access_token_duration + 1))
            self.post(url, headers=headers, body=body, expected_status=http.client.UNAUTHORIZED)

    def test_missing_oauth_headers(self):
        endpoint = '/OS-OAUTH1/request_token'
        client = oauth1.Client(uuid.uuid4().hex, client_secret=uuid.uuid4().hex, signature_method=oauth1.SIG_HMAC, callback_uri='oob')
        headers = {'requested_project_id': uuid.uuid4().hex}
        _url, headers, _body = client.sign(self.base_url + endpoint, http_method='POST', headers=headers)
        del headers['Authorization']
        self.post(endpoint, headers=headers, expected_status=http.client.INTERNAL_SERVER_ERROR)