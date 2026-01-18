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
class TestIdentityProxyGroup(TestIdentityProxyBase):

    def test_group_create_attrs(self):
        self.verify_create(self.proxy.create_group, group.Group)

    def test_group_delete(self):
        self.verify_delete(self.proxy.delete_group, group.Group, False)

    def test_group_delete_ignore(self):
        self.verify_delete(self.proxy.delete_group, group.Group, True)

    def test_group_find(self):
        self.verify_find(self.proxy.find_group, group.Group)

    def test_group_get(self):
        self.verify_get(self.proxy.get_group, group.Group)

    def test_groups(self):
        self.verify_list(self.proxy.groups, group.Group)

    def test_group_update(self):
        self.verify_update(self.proxy.update_group, group.Group)

    def test_add_user_to_group(self):
        self._verify('openstack.identity.v3.group.Group.add_user', self.proxy.add_user_to_group, method_args=['uid', 'gid'], expected_args=[self.proxy, self.proxy._get_resource(user.User, 'uid')])

    def test_remove_user_from_group(self):
        self._verify('openstack.identity.v3.group.Group.remove_user', self.proxy.remove_user_from_group, method_args=['uid', 'gid'], expected_args=[self.proxy, self.proxy._get_resource(user.User, 'uid')])

    def test_check_user_in_group(self):
        self._verify('openstack.identity.v3.group.Group.check_user', self.proxy.check_user_in_group, method_args=['uid', 'gid'], expected_args=[self.proxy, self.proxy._get_resource(user.User, 'uid')])

    def test_group_users(self):
        self.verify_list(self.proxy.group_users, user.User, method_kwargs={'group': 'group', 'attrs': 1}, expected_kwargs={'attrs': 1})