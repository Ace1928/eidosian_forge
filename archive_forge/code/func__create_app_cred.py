from base64 import b64encode
from cryptography.hazmat.primitives.serialization import Encoding
import fixtures
import http
from http import client
from oslo_log import log
from oslo_serialization import jsonutils
from unittest import mock
from urllib import parse
from keystone.api.os_oauth2 import AccessTokenResource
from keystone.common import provider_api
from keystone.common import utils
from keystone import conf
from keystone import exception
from keystone.federation.utils import RuleProcessor
from keystone.tests import unit
from keystone.tests.unit import test_v3
from keystone.token.provider import Manager
def _create_app_cred(self, user_id, app_cred_name):
    resp = self.post(self.APP_CRED_CREATE_URL % {'user_id': user_id}, body={'application_credential': {'name': app_cred_name}})
    LOG.debug(f'resp: {resp}')
    app_ref = resp.result['application_credential']
    return app_ref