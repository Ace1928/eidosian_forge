import datetime
import io
import itertools
import json
import logging
import sys
from unittest import mock
import uuid
from oslo_utils import encodeutils
import requests
import requests.auth
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import plugin
from keystoneauth1 import session as client_session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
class RedirectTests(utils.TestCase):
    REDIRECT_CHAIN = ['http://myhost:3445/', 'http://anotherhost:6555/', 'http://thirdhost/', 'http://finaldestination:55/']
    DEFAULT_REDIRECT_BODY = 'Redirect'
    DEFAULT_RESP_BODY = 'Found'

    def setup_redirects(self, method='GET', status_code=305, redirect_kwargs={}, final_kwargs={}):
        redirect_kwargs.setdefault('text', self.DEFAULT_REDIRECT_BODY)
        for s, d in zip(self.REDIRECT_CHAIN, self.REDIRECT_CHAIN[1:]):
            self.requests_mock.register_uri(method, s, status_code=status_code, headers={'Location': d}, **redirect_kwargs)
        final_kwargs.setdefault('status_code', 200)
        final_kwargs.setdefault('text', self.DEFAULT_RESP_BODY)
        self.requests_mock.register_uri(method, self.REDIRECT_CHAIN[-1], **final_kwargs)

    def assertResponse(self, resp):
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.text, self.DEFAULT_RESP_BODY)

    def test_basic_get(self):
        session = client_session.Session()
        self.setup_redirects()
        resp = session.get(self.REDIRECT_CHAIN[-2])
        self.assertResponse(resp)

    def test_basic_post_keeps_correct_method(self):
        session = client_session.Session()
        self.setup_redirects(method='POST', status_code=301)
        resp = session.post(self.REDIRECT_CHAIN[-2])
        self.assertResponse(resp)

    def test_redirect_forever(self):
        session = client_session.Session(redirect=True)
        self.setup_redirects()
        resp = session.get(self.REDIRECT_CHAIN[0])
        self.assertResponse(resp)
        self.assertTrue(len(resp.history), len(self.REDIRECT_CHAIN))

    def test_no_redirect(self):
        session = client_session.Session(redirect=False)
        self.setup_redirects()
        resp = session.get(self.REDIRECT_CHAIN[0])
        self.assertEqual(resp.status_code, 305)
        self.assertEqual(resp.url, self.REDIRECT_CHAIN[0])

    def test_redirect_limit(self):
        self.setup_redirects()
        for i in (1, 2):
            session = client_session.Session(redirect=i)
            resp = session.get(self.REDIRECT_CHAIN[0])
            self.assertEqual(resp.status_code, 305)
            self.assertEqual(resp.url, self.REDIRECT_CHAIN[i])
            self.assertEqual(resp.text, self.DEFAULT_REDIRECT_BODY)

    def test_redirect_with_params(self):
        params = {'foo': 'bar'}
        session = client_session.Session(redirect=True)
        self.setup_redirects(final_kwargs={'complete_qs': True})
        resp = session.get(self.REDIRECT_CHAIN[0], params=params)
        self.assertResponse(resp)
        self.assertTrue(len(resp.history), len(self.REDIRECT_CHAIN))
        self.assertQueryStringIs(None)

    def test_history_matches_requests(self):
        self.setup_redirects(status_code=301)
        session = client_session.Session(redirect=True)
        req_resp = requests.get(self.REDIRECT_CHAIN[0], allow_redirects=True)
        ses_resp = session.get(self.REDIRECT_CHAIN[0])
        self.assertEqual(len(req_resp.history), len(ses_resp.history))
        for r, s in zip(req_resp.history, ses_resp.history):
            self.assertEqual(r.url, s.url)
            self.assertEqual(r.status_code, s.status_code)

    def test_permanent_redirect_308(self):
        session = client_session.Session()
        self.setup_redirects(status_code=308)
        resp = session.get(self.REDIRECT_CHAIN[-2])
        self.assertResponse(resp)

    def test_req_id_redirect(self):
        session = client_session.Session()
        self.setup_redirects(status_code=302)
        resp = session.get(self.REDIRECT_CHAIN[0], headers={'x-openstack-request-id': 'req-1234-5678'})
        self.assertResponse(resp)
        self.assertRequestHeaderEqual('x-openstack-request-id', 'req-1234-5678')