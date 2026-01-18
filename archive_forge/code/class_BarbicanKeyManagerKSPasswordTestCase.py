import abc
from keystoneauth1 import identity
from keystoneauth1 import session
from oslo_config import cfg
from oslo_context import context
from oslo_utils import uuidutils
from oslotest import base
from testtools import testcase
from castellan.common.credentials import keystone_password
from castellan.common.credentials import keystone_token
from castellan.common import exception
from castellan.key_manager import barbican_key_manager
from castellan.tests.functional import config
from castellan.tests.functional.key_manager import test_key_manager
from castellan.tests import utils
class BarbicanKeyManagerKSPasswordTestCase(BarbicanKeyManagerTestCase, base.BaseTestCase):

    def get_context(self):
        auth_url = CONF.identity.auth_url
        username = CONF.identity.username
        password = CONF.identity.password
        project_name = CONF.identity.project_name
        user_domain_name = CONF.identity.user_domain_name
        project_domain_name = CONF.identity.project_domain_name
        ctxt = keystone_password.KeystonePassword(auth_url=auth_url, username=username, password=password, project_name=project_name, user_domain_name=user_domain_name, project_domain_name=project_domain_name)
        return ctxt