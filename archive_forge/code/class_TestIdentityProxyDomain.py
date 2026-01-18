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
class TestIdentityProxyDomain(TestIdentityProxyBase):

    def test_domain_create_attrs(self):
        self.verify_create(self.proxy.create_domain, domain.Domain)

    def test_domain_delete(self):
        self.verify_delete(self.proxy.delete_domain, domain.Domain, False)

    def test_domain_delete_ignore(self):
        self.verify_delete(self.proxy.delete_domain, domain.Domain, True)

    def test_domain_find(self):
        self.verify_find(self.proxy.find_domain, domain.Domain)

    def test_domain_get(self):
        self.verify_get(self.proxy.get_domain, domain.Domain)

    def test_domains(self):
        self.verify_list(self.proxy.domains, domain.Domain)

    def test_domain_update(self):
        self.verify_update(self.proxy.update_domain, domain.Domain)