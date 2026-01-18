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
def _connect_retries_check(self, session, expected_retries=0, call_args=None):
    call_args = call_args or {}
    self.stub_url('GET', exc=requests.exceptions.Timeout())
    call_args['url'] = self.TEST_URL
    with mock.patch('time.sleep') as m:
        self.assertRaises(exceptions.ConnectTimeout, session.get, **call_args)
        self.assertEqual(expected_retries, m.call_count)
        m.assert_called_with(2.0)
    self.assertThat(self.requests_mock.request_history, matchers.HasLength(expected_retries + 1))