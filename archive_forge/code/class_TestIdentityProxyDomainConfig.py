import uuid
from openstack.identity.v3 import _proxy
from openstack.identity.v3 import access_rule
from openstack.identity.v3 import credential
from openstack.identity.v3 import domain
from openstack.identity.v3 import domain_config
from openstack.identity.v3 import endpoint
from openstack.identity.v3 import group
from openstack.identity.v3 import policy
from openstack.identity.v3 import project
from openstack.identity.v3 import region
from openstack.identity.v3 import role
from openstack.identity.v3 import role_domain_group_assignment
from openstack.identity.v3 import role_domain_user_assignment
from openstack.identity.v3 import role_project_group_assignment
from openstack.identity.v3 import role_project_user_assignment
from openstack.identity.v3 import role_system_group_assignment
from openstack.identity.v3 import role_system_user_assignment
from openstack.identity.v3 import service
from openstack.identity.v3 import trust
from openstack.identity.v3 import user
from openstack.tests.unit import test_proxy_base
class TestIdentityProxyDomainConfig(TestIdentityProxyBase):

    def test_domain_config_create_attrs(self):
        self.verify_create(self.proxy.create_domain_config, domain_config.DomainConfig, method_args=['domain_id'], method_kwargs={}, expected_args=[], expected_kwargs={'domain_id': 'domain_id'})

    def test_domain_config_delete(self):
        self.verify_delete(self.proxy.delete_domain_config, domain_config.DomainConfig, ignore_missing=False, method_args=['domain_id'], method_kwargs={}, expected_args=[], expected_kwargs={'domain_id': 'domain_id'})

    def test_domain_config_delete_ignore(self):
        self.verify_delete(self.proxy.delete_domain_config, domain_config.DomainConfig, ignore_missing=True, method_args=['domain_id'], method_kwargs={}, expected_args=[], expected_kwargs={'domain_id': 'domain_id'})

    def test_domain_config_get(self):
        self.verify_get(self.proxy.get_domain_config, domain_config.DomainConfig, method_args=['domain_id'], method_kwargs={}, expected_args=[], expected_kwargs={'domain_id': 'domain_id', 'requires_id': False})

    def test_domain_config_update(self):
        self.verify_update(self.proxy.update_domain_config, domain_config.DomainConfig, method_args=['domain_id'], method_kwargs={}, expected_args=[], expected_kwargs={'domain_id': 'domain_id'})