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
def _create_project_user(self, no_roles=False):
    new_domain_ref = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(new_domain_ref['id'], new_domain_ref)
    new_project_ref = unit.new_project_ref(domain_id=self.domain_id)
    PROVIDERS.resource_api.create_project(new_project_ref['id'], new_project_ref)
    new_user = unit.create_user(PROVIDERS.identity_api, domain_id=new_domain_ref['id'], project_id=new_project_ref['id'])
    if not no_roles:
        PROVIDERS.assignment_api.create_grant(self.role['id'], user_id=new_user['id'], project_id=new_project_ref['id'])
    return (new_user, new_domain_ref, new_project_ref)