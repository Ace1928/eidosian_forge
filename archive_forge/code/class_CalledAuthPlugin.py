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
class CalledAuthPlugin(plugin.BaseAuthPlugin):
    ENDPOINT = 'http://fakeendpoint/'
    TOKEN = utils.TestCase.TEST_TOKEN
    USER_ID = uuid.uuid4().hex
    PROJECT_ID = uuid.uuid4().hex

    def __init__(self, invalidate=True):
        self.get_token_called = False
        self.get_endpoint_called = False
        self.endpoint_arguments = {}
        self.invalidate_called = False
        self.get_project_id_called = False
        self.get_user_id_called = False
        self._invalidate = invalidate

    def get_token(self, session):
        self.get_token_called = True
        return self.TOKEN

    def get_endpoint(self, session, **kwargs):
        self.get_endpoint_called = True
        self.endpoint_arguments = kwargs
        return self.ENDPOINT

    def invalidate(self):
        self.invalidate_called = True
        return self._invalidate

    def get_project_id(self, session, **kwargs):
        self.get_project_id_called = True
        return self.PROJECT_ID

    def get_user_id(self, session, **kwargs):
        self.get_user_id_called = True
        return self.USER_ID