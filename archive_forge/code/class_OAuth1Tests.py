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
class OAuth1Tests(test_v3.RestfulTestCase):
    CONSUMER_URL = '/OS-OAUTH1/consumers'

    def setUp(self):
        super(OAuth1Tests, self).setUp()
        self.base_url = 'http://localhost/v3'

    def _create_single_consumer(self):
        ref = {'description': uuid.uuid4().hex}
        resp = self.post(self.CONSUMER_URL, body={'consumer': ref})
        return resp.result['consumer']

    def _create_request_token(self, consumer, project_id, base_url=None):
        endpoint = '/OS-OAUTH1/request_token'
        client = oauth1.Client(consumer['key'], client_secret=consumer['secret'], signature_method=oauth1.SIG_HMAC, callback_uri='oob')
        headers = {'requested_project_id': project_id}
        if not base_url:
            base_url = self.base_url
        url, headers, body = client.sign(base_url + endpoint, http_method='POST', headers=headers)
        return (endpoint, headers)

    def _create_access_token(self, consumer, token, base_url=None):
        endpoint = '/OS-OAUTH1/access_token'
        client = oauth1.Client(consumer['key'], client_secret=consumer['secret'], resource_owner_key=token.key, resource_owner_secret=token.secret, signature_method=oauth1.SIG_HMAC, verifier=token.verifier)
        if not base_url:
            base_url = self.base_url
        url, headers, body = client.sign(base_url + endpoint, http_method='POST')
        headers.update({'Content-Type': 'application/json'})
        return (endpoint, headers)

    def _get_oauth_token(self, consumer, token):
        client = oauth1.Client(consumer['key'], client_secret=consumer['secret'], resource_owner_key=token.key, resource_owner_secret=token.secret, signature_method=oauth1.SIG_HMAC)
        endpoint = '/auth/tokens'
        url, headers, body = client.sign(self.base_url + endpoint, http_method='POST')
        headers.update({'Content-Type': 'application/json'})
        ref = {'auth': {'identity': {'oauth1': {}, 'methods': ['oauth1']}}}
        return (endpoint, headers, ref)

    def _authorize_request_token(self, request_id):
        if isinstance(request_id, bytes):
            request_id = request_id.decode()
        return '/OS-OAUTH1/authorize/%s' % request_id