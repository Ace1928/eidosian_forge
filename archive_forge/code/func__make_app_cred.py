import copy
import datetime
import fixtures
import itertools
import operator
import re
from unittest import mock
from urllib import parse
import uuid
from cryptography.hazmat.primitives.serialization import Encoding
import freezegun
import http.client
from oslo_serialization import jsonutils as json
from oslo_utils import fixture
from oslo_utils import timeutils
from testtools import matchers
from testtools import testcase
from keystone import auth
from keystone.auth.plugins import totp
from keystone.common import authorization
from keystone.common import provider_api
from keystone.common.rbac_enforcer import policy
from keystone.common import utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import resource_options as ro
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
def _make_app_cred(self, expires=None, access_rules=None):
    roles = [{'id': self.role_id}]
    data = {'id': uuid.uuid4().hex, 'name': uuid.uuid4().hex, 'secret': uuid.uuid4().hex, 'user_id': self.user['id'], 'project_id': self.project['id'], 'description': uuid.uuid4().hex, 'roles': roles}
    if expires:
        data['expires_at'] = expires
    if access_rules:
        data['access_rules'] = access_rules
    return data