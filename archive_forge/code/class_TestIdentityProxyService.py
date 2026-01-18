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
class TestIdentityProxyService(TestIdentityProxyBase):

    def test_service_create_attrs(self):
        self.verify_create(self.proxy.create_service, service.Service)

    def test_service_delete(self):
        self.verify_delete(self.proxy.delete_service, service.Service, False)

    def test_service_delete_ignore(self):
        self.verify_delete(self.proxy.delete_service, service.Service, True)

    def test_service_find(self):
        self.verify_find(self.proxy.find_service, service.Service)

    def test_service_get(self):
        self.verify_get(self.proxy.get_service, service.Service)

    def test_services(self):
        self.verify_list(self.proxy.services, service.Service)

    def test_service_update(self):
        self.verify_update(self.proxy.update_service, service.Service)