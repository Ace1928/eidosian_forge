from unittest import mock
from openstack.shared_file_system.v2 import _proxy
from openstack.shared_file_system.v2 import limit
from openstack.shared_file_system.v2 import resource_locks
from openstack.shared_file_system.v2 import share
from openstack.shared_file_system.v2 import share_access_rule
from openstack.shared_file_system.v2 import share_group
from openstack.shared_file_system.v2 import share_group_snapshot
from openstack.shared_file_system.v2 import share_instance
from openstack.shared_file_system.v2 import share_network
from openstack.shared_file_system.v2 import share_network_subnet
from openstack.shared_file_system.v2 import share_snapshot
from openstack.shared_file_system.v2 import share_snapshot_instance
from openstack.shared_file_system.v2 import storage_pool
from openstack.shared_file_system.v2 import user_message
from openstack.tests.unit import test_proxy_base
class TestAccessRuleProxy(test_proxy_base.TestProxyBase):

    def setUp(self):
        super(TestAccessRuleProxy, self).setUp()
        self.proxy = _proxy.Proxy(self.session)

    def test_access_ruless(self):
        self.verify_list(self.proxy.access_rules, share_access_rule.ShareAccessRule, method_args=['test_share'], expected_args=[], expected_kwargs={'share_id': 'test_share'})

    def test_access_rules_get(self):
        self.verify_get(self.proxy.get_access_rule, share_access_rule.ShareAccessRule)

    def test_access_rules_create(self):
        self.verify_create(self.proxy.create_access_rule, share_access_rule.ShareAccessRule, method_args=['share_id'], expected_args=[])

    def test_access_rules_delete(self):
        self._verify('openstack.shared_file_system.v2.share_access_rule.' + 'ShareAccessRule.delete', self.proxy.delete_access_rule, method_args=['access_id', 'share_id', 'ignore_missing'], expected_args=[self.proxy, 'share_id'], expected_kwargs={'unrestrict': False})