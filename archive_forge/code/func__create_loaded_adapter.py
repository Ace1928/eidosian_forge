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
def _create_loaded_adapter(self, sess=None, auth=None):
    return adapter.Adapter(sess or client_session.Session(), auth=auth or CalledAuthPlugin(), service_type=self.SERVICE_TYPE, service_name=self.SERVICE_NAME, interface=self.INTERFACE, region_name=self.REGION_NAME, user_agent=self.USER_AGENT, version=self.VERSION, allow=self.ALLOW)