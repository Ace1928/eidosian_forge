from unittest import mock
from openstack.block_storage.v3 import _proxy
from openstack.block_storage.v3 import backup
from openstack.block_storage.v3 import capabilities
from openstack.block_storage.v3 import extension
from openstack.block_storage.v3 import group
from openstack.block_storage.v3 import group_snapshot
from openstack.block_storage.v3 import group_type
from openstack.block_storage.v3 import limits
from openstack.block_storage.v3 import quota_set
from openstack.block_storage.v3 import resource_filter
from openstack.block_storage.v3 import service
from openstack.block_storage.v3 import snapshot
from openstack.block_storage.v3 import stats
from openstack.block_storage.v3 import type
from openstack.block_storage.v3 import volume
from openstack import resource
from openstack.tests.unit import test_proxy_base
class TestGroupSnapshot(TestVolumeProxy):

    def test_group_snapshot_get(self):
        self.verify_get(self.proxy.get_group_snapshot, group_snapshot.GroupSnapshot)

    def test_group_snapshot_find(self):
        self.verify_find(self.proxy.find_group_snapshot, group_snapshot.GroupSnapshot, expected_kwargs={'list_base_path': '/group_snapshots/detail'})

    def test_group_snapshots(self):
        self.verify_list(self.proxy.group_snapshots, group_snapshot.GroupSnapshot, expected_kwargs={})

    def test_group_snapshots__detailed(self):
        self.verify_list(self.proxy.group_snapshots, group_snapshot.GroupSnapshot, method_kwargs={'details': True, 'query': 1}, expected_kwargs={'query': 1, 'base_path': '/group_snapshots/detail'})

    def test_group_snapshot_create(self):
        self.verify_create(self.proxy.create_group_snapshot, group_snapshot.GroupSnapshot)

    def test_group_snapshot_delete(self):
        self.verify_delete(self.proxy.delete_group_snapshot, group_snapshot.GroupSnapshot, False)

    def test_group_snapshot_delete_ignore(self):
        self.verify_delete(self.proxy.delete_group_snapshot, group_snapshot.GroupSnapshot, True)