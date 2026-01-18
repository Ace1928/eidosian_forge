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
class TestIdentityProxyTrust(TestIdentityProxyBase):

    def test_trust_create_attrs(self):
        self.verify_create(self.proxy.create_trust, trust.Trust)

    def test_trust_delete(self):
        self.verify_delete(self.proxy.delete_trust, trust.Trust, False)

    def test_trust_delete_ignore(self):
        self.verify_delete(self.proxy.delete_trust, trust.Trust, True)

    def test_trust_find(self):
        self.verify_find(self.proxy.find_trust, trust.Trust)

    def test_trust_get(self):
        self.verify_get(self.proxy.get_trust, trust.Trust)

    def test_trusts(self):
        self.verify_list(self.proxy.trusts, trust.Trust)