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
class KeystoneClientTestDomainName(KeystoneClientTest):

    def setUp(self):
        cfg.CONF.set_override('stack_user_domain_name', 'fake_domain_name')
        super(KeystoneClientTestDomainName, self).setUp()
        cfg.CONF.clear_override('stack_user_domain_id')

    def _clear_domain_override(self):
        cfg.CONF.clear_override('stack_user_domain_name')

    def _validate_stub_domain_admin_client(self):
        ks_auth.Password.assert_called_once_with(auth_url='http://server.test:5000/v3', password='adminsecret', domain_id=None, domain_name='fake_domain_name', user_domain_id=None, user_domain_name='fake_domain_name', username='adminuser123')
        self.m_client.assert_called_once_with(session=utils.AnyInstance(ks_session.Session), auth=self.mock_ks_auth, connect_retries=2, interface='publicURL', region_name=None)

    def _stub_domain_admin_client(self, domain_id='adomain123'):
        super(KeystoneClientTestDomainName, self)._stub_domain_admin_client()
        if domain_id:
            self.mock_ks_auth.get_access.return_value.domain_id = domain_id

    def test_enable_stack_domain_user_error_project(self):
        p = super(KeystoneClientTestDomainName, self)
        p.test_enable_stack_domain_user_error_project()

    def test_delete_stack_domain_user_keypair(self):
        p = super(KeystoneClientTestDomainName, self)
        p.test_delete_stack_domain_user_keypair()

    def test_delete_stack_domain_user_error_project(self):
        p = super(KeystoneClientTestDomainName, self)
        p.test_delete_stack_domain_user_error_project()

    def test_delete_stack_domain_user_keypair_error_project(self):
        p = super(KeystoneClientTestDomainName, self)
        p.test_delete_stack_domain_user_keypair_error_project()

    def test_delete_stack_domain_user(self):
        p = super(KeystoneClientTestDomainName, self)
        p.test_delete_stack_domain_user()

    def test_enable_stack_domain_user(self):
        p = super(KeystoneClientTestDomainName, self)
        p.test_enable_stack_domain_user()

    def test_delete_stack_domain_user_error_domain(self):
        p = super(KeystoneClientTestDomainName, self)
        p.test_delete_stack_domain_user_error_domain()

    def test_disable_stack_domain_user_error_project(self):
        p = super(KeystoneClientTestDomainName, self)
        p.test_disable_stack_domain_user_error_project()

    def test_enable_stack_domain_user_error_domain(self):
        p = super(KeystoneClientTestDomainName, self)
        p.test_enable_stack_domain_user_error_domain()

    def test_delete_stack_domain_user_keypair_error_domain(self):
        p = super(KeystoneClientTestDomainName, self)
        p.test_delete_stack_domain_user_keypair_error_domain()

    def test_disable_stack_domain_user(self):
        p = super(KeystoneClientTestDomainName, self)
        p.test_disable_stack_domain_user()

    def test_disable_stack_domain_user_error_domain(self):
        p = super(KeystoneClientTestDomainName, self)
        p.test_disable_stack_domain_user_error_domain()

    def test_delete_stack_domain_project(self):
        p = super(KeystoneClientTestDomainName, self)
        p.test_delete_stack_domain_project()

    def test_delete_stack_domain_project_notfound(self):
        p = super(KeystoneClientTestDomainName, self)
        p.test_delete_stack_domain_project_notfound()

    def test_delete_stack_domain_project_wrongdomain(self):
        p = super(KeystoneClientTestDomainName, self)
        p.test_delete_stack_domain_project_wrongdomain()

    def test_create_stack_domain_project(self):
        p = super(KeystoneClientTestDomainName, self)
        p.test_create_stack_domain_project()

    def test_create_stack_domain_user(self):
        p = super(KeystoneClientTestDomainName, self)
        p.test_create_stack_domain_user()