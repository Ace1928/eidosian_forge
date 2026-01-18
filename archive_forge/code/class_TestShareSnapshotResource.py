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
class TestShareSnapshotResource(test_proxy_base.TestProxyBase):

    def setUp(self):
        super(TestShareSnapshotResource, self).setUp()
        self.proxy = _proxy.Proxy(self.session)

    def test_share_snapshots(self):
        self.verify_list(self.proxy.share_snapshots, share_snapshot.ShareSnapshot)

    def test_share_snapshots_detailed(self):
        self.verify_list(self.proxy.share_snapshots, share_snapshot.ShareSnapshot, method_kwargs={'details': True, 'name': 'my_snapshot'}, expected_kwargs={'name': 'my_snapshot'})

    def test_share_snapshots_not_detailed(self):
        self.verify_list(self.proxy.share_snapshots, share_snapshot.ShareSnapshot, method_kwargs={'details': False, 'name': 'my_snapshot'}, expected_kwargs={'name': 'my_snapshot'})

    def test_share_snapshot_get(self):
        self.verify_get(self.proxy.get_share_snapshot, share_snapshot.ShareSnapshot)

    def test_share_snapshot_delete(self):
        self.verify_delete(self.proxy.delete_share_snapshot, share_snapshot.ShareSnapshot, False)

    def test_share_snapshot_delete_ignore(self):
        self.verify_delete(self.proxy.delete_share_snapshot, share_snapshot.ShareSnapshot, True)

    def test_share_snapshot_create(self):
        self.verify_create(self.proxy.create_share_snapshot, share_snapshot.ShareSnapshot)

    def test_share_snapshot_update(self):
        self.verify_update(self.proxy.update_share_snapshot, share_snapshot.ShareSnapshot)

    @mock.patch('openstack.resource.wait_for_delete')
    def test_wait_for_delete(self, mock_wait):
        mock_resource = mock.Mock()
        mock_wait.return_value = mock_resource
        self.proxy.wait_for_delete(mock_resource)
        mock_wait.assert_called_once_with(self.proxy, mock_resource, 2, 120)