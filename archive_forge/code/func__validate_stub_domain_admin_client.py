import json
from unittest import mock
import uuid
from keystoneauth1 import access as ks_access
from keystoneauth1 import exceptions as kc_exception
from keystoneauth1.identity import access as ks_auth_access
from keystoneauth1.identity import generic as ks_auth
from keystoneauth1 import loading as ks_loading
from keystoneauth1 import session as ks_session
from keystoneauth1 import token_endpoint as ks_token_endpoint
from keystoneclient.v3 import client as kc_v3
from keystoneclient.v3 import domains as kc_v3_domains
from oslo_config import cfg
from heat.common import config
from heat.common import exception
from heat.common import password_gen
from heat.engine.clients.os.keystone import heat_keystoneclient
from heat.tests import common
from heat.tests import utils
def _validate_stub_domain_admin_client(self):
    ks_auth.Password.assert_called_once_with(auth_url='http://server.test:5000/v3', password='adminsecret', domain_id=None, domain_name='fake_domain_name', user_domain_id=None, user_domain_name='fake_domain_name', username='adminuser123')
    self.m_client.assert_called_once_with(session=utils.AnyInstance(ks_session.Session), auth=self.mock_ks_auth, connect_retries=2, interface='publicURL', region_name=None)