from unittest import mock
from openstack.block_storage.v2 import _proxy
from openstack.block_storage.v2 import backup
from openstack.block_storage.v2 import capabilities
from openstack.block_storage.v2 import limits
from openstack.block_storage.v2 import quota_set
from openstack.block_storage.v2 import snapshot
from openstack.block_storage.v2 import stats
from openstack.block_storage.v2 import type
from openstack.block_storage.v2 import volume
from openstack import resource
from openstack.tests.unit import test_proxy_base
class TestVolumeActions(TestVolumeProxy):

    def test_volume_extend(self):
        self._verify('openstack.block_storage.v2.volume.Volume.extend', self.proxy.extend_volume, method_args=['value', 'new-size'], expected_args=[self.proxy, 'new-size'])

    def test_volume_set_bootable(self):
        self._verify('openstack.block_storage.v2.volume.Volume.set_bootable_status', self.proxy.set_volume_bootable_status, method_args=['value', True], expected_args=[self.proxy, True])

    def test_volume_reset_volume_status(self):
        self._verify('openstack.block_storage.v2.volume.Volume.reset_status', self.proxy.reset_volume_status, method_args=['value', '1', '2', '3'], expected_args=[self.proxy, '1', '2', '3'])

    def test_attach_instance(self):
        self._verify('openstack.block_storage.v2.volume.Volume.attach', self.proxy.attach_volume, method_args=['value', '1'], method_kwargs={'instance': '2'}, expected_args=[self.proxy, '1', '2', None])

    def test_attach_host(self):
        self._verify('openstack.block_storage.v2.volume.Volume.attach', self.proxy.attach_volume, method_args=['value', '1'], method_kwargs={'host_name': '3'}, expected_args=[self.proxy, '1', None, '3'])

    def test_detach_defaults(self):
        self._verify('openstack.block_storage.v2.volume.Volume.detach', self.proxy.detach_volume, method_args=['value', '1'], expected_args=[self.proxy, '1', False, None])

    def test_detach_force(self):
        self._verify('openstack.block_storage.v2.volume.Volume.detach', self.proxy.detach_volume, method_args=['value', '1', True, {'a': 'b'}], expected_args=[self.proxy, '1', True, {'a': 'b'}])

    def test_unmanage(self):
        self._verify('openstack.block_storage.v2.volume.Volume.unmanage', self.proxy.unmanage_volume, method_args=['value'], expected_args=[self.proxy])

    def test_migrate_default(self):
        self._verify('openstack.block_storage.v2.volume.Volume.migrate', self.proxy.migrate_volume, method_args=['value', '1'], expected_args=[self.proxy, '1', False, False])

    def test_migrate_nondefault(self):
        self._verify('openstack.block_storage.v2.volume.Volume.migrate', self.proxy.migrate_volume, method_args=['value', '1', True, True], expected_args=[self.proxy, '1', True, True])

    def test_complete_migration(self):
        self._verify('openstack.block_storage.v2.volume.Volume.complete_migration', self.proxy.complete_volume_migration, method_args=['value', '1'], expected_args=[self.proxy, '1', False])

    def test_complete_migration_error(self):
        self._verify('openstack.block_storage.v2.volume.Volume.complete_migration', self.proxy.complete_volume_migration, method_args=['value', '1', True], expected_args=[self.proxy, '1', True])