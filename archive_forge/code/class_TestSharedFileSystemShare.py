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
class TestSharedFileSystemShare(TestSharedFileSystemProxy):

    def test_shares(self):
        self.verify_list(self.proxy.shares, share.Share)

    def test_shares_detailed(self):
        self.verify_list(self.proxy.shares, share.Share, method_kwargs={'details': True, 'query': 1}, expected_kwargs={'query': 1})

    def test_shares_not_detailed(self):
        self.verify_list(self.proxy.shares, share.Share, method_kwargs={'details': False, 'query': 1}, expected_kwargs={'query': 1})

    def test_share_get(self):
        self.verify_get(self.proxy.get_share, share.Share)

    def test_share_find(self):
        self.verify_find(self.proxy.find_share, share.Share)

    def test_share_delete(self):
        self.verify_delete(self.proxy.delete_share, share.Share, False)

    def test_share_delete_ignore(self):
        self.verify_delete(self.proxy.delete_share, share.Share, True)

    def test_share_create(self):
        self.verify_create(self.proxy.create_share, share.Share)

    def test_share_update(self):
        self.verify_update(self.proxy.update_share, share.Share)

    def test_share_resize_extend(self):
        mock_share = share.Share(size=10, id='fakeId')
        self.proxy._get = mock.Mock(return_value=mock_share)
        self._verify('openstack.shared_file_system.v2.share.' + 'Share.extend_share', self.proxy.resize_share, method_args=['fakeId', 20], expected_args=[self.proxy, 20, False])

    def test_share_resize_shrink(self):
        mock_share = share.Share(size=30, id='fakeId')
        self.proxy._get = mock.Mock(return_value=mock_share)
        self._verify('openstack.shared_file_system.v2.share.' + 'Share.shrink_share', self.proxy.resize_share, method_args=['fakeId', 20], expected_args=[self.proxy, 20])

    def test_share_instances(self):
        self.verify_list(self.proxy.share_instances, share_instance.ShareInstance)

    def test_share_instance_get(self):
        self.verify_get(self.proxy.get_share_instance, share_instance.ShareInstance)

    def test_share_instance_reset(self):
        self._verify('openstack.shared_file_system.v2.share_instance.' + 'ShareInstance.reset_status', self.proxy.reset_share_instance_status, method_args=['id', 'available'], expected_args=[self.proxy, 'available'])

    def test_share_instance_delete(self):
        self._verify('openstack.shared_file_system.v2.share_instance.' + 'ShareInstance.force_delete', self.proxy.delete_share_instance, method_args=['id'], expected_args=[self.proxy])

    @mock.patch('openstack.resource.wait_for_status')
    def test_wait_for(self, mock_wait):
        mock_resource = mock.Mock()
        mock_wait.return_value = mock_resource
        self.proxy.wait_for_status(mock_resource, 'ACTIVE')
        mock_wait.assert_called_once_with(self.proxy, mock_resource, 'ACTIVE', [], 2, 120, attribute='status')