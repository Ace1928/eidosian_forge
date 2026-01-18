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
class TestUserMessageProxy(test_proxy_base.TestProxyBase):

    def setUp(self):
        super(TestUserMessageProxy, self).setUp()
        self.proxy = _proxy.Proxy(self.session)

    def test_user_messages(self):
        self.verify_list(self.proxy.user_messages, user_message.UserMessage)

    def test_user_messages_queried(self):
        self.verify_list(self.proxy.user_messages, user_message.UserMessage, method_kwargs={'action_id': '1'}, expected_kwargs={'action_id': '1'})

    def test_user_message_get(self):
        self.verify_get(self.proxy.get_user_message, user_message.UserMessage)

    def test_delete_user_message(self):
        self.verify_delete(self.proxy.delete_user_message, user_message.UserMessage, False)

    def test_delete_user_message_true(self):
        self.verify_delete(self.proxy.delete_user_message, user_message.UserMessage, True)

    def test_limit(self):
        self.verify_list(self.proxy.limits, limit.Limit)