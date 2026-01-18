from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3.volumes import Volume
class VolumesTest(utils.TestCase):
    """Block Storage API v3.0"""

    def test_list_volumes_with_marker_limit(self):
        lst = cs.volumes.list(marker=1234, limit=2)
        cs.assert_called('GET', '/volumes/detail?limit=2&marker=1234')
        self._assert_request_id(lst)

    def test__list(self):
        limit = 2
        url = '/volumes?limit=%s' % limit
        response_key = 'volumes'
        fake_volume1234 = Volume(self, {'id': 1234, 'name': 'sample-volume'}, loaded=True)
        fake_volume5678 = Volume(self, {'id': 5678, 'name': 'sample-volume2'}, loaded=True)
        fake_volumes = [fake_volume1234, fake_volume5678]
        volumes = cs.volumes._list(url, response_key, limit=limit)
        self._assert_request_id(volumes)
        cs.assert_called('GET', url)
        self.assertEqual(fake_volumes, volumes)
        cs.client.osapi_max_limit = 1
        volumes = cs.volumes._list(url, response_key, limit=limit)
        self.assertEqual(fake_volumes, volumes)
        self._assert_request_id(volumes)
        cs.client.osapi_max_limit = 1000

    def test_create_volume(self):
        vol = cs.volumes.create(1)
        cs.assert_called('POST', '/volumes')
        self._assert_request_id(vol)

    def test_delete_volume(self):
        v = cs.volumes.list()[0]
        del_v = v.delete()
        cs.assert_called('DELETE', '/volumes/1234')
        self._assert_request_id(del_v)
        del_v = cs.volumes.delete('1234')
        cs.assert_called('DELETE', '/volumes/1234')
        self._assert_request_id(del_v)
        del_v = cs.volumes.delete(v)
        cs.assert_called('DELETE', '/volumes/1234')
        self._assert_request_id(del_v)

    def test_attach(self):
        v = cs.volumes.get('1234')
        self._assert_request_id(v)
        vol = cs.volumes.attach(v, 1, '/dev/vdc', mode='ro')
        cs.assert_called('POST', '/volumes/1234/action')
        self._assert_request_id(vol)

    def test_attach_to_host(self):
        v = cs.volumes.get('1234')
        self._assert_request_id(v)
        vol = cs.volumes.attach(v, None, None, host_name='test', mode='rw')
        cs.assert_called('POST', '/volumes/1234/action')
        self._assert_request_id(vol)

    def test_detach(self):
        v = cs.volumes.get('1234')
        self._assert_request_id(v)
        vol = cs.volumes.detach(v, 'abc123')
        cs.assert_called('POST', '/volumes/1234/action')
        self._assert_request_id(vol)

    def test_reserve(self):
        v = cs.volumes.get('1234')
        self._assert_request_id(v)
        vol = cs.volumes.reserve(v)
        cs.assert_called('POST', '/volumes/1234/action')
        self._assert_request_id(vol)

    def test_unreserve(self):
        v = cs.volumes.get('1234')
        self._assert_request_id(v)
        vol = cs.volumes.unreserve(v)
        cs.assert_called('POST', '/volumes/1234/action')
        self._assert_request_id(vol)

    def test_begin_detaching(self):
        v = cs.volumes.get('1234')
        cs.volumes.begin_detaching(v)
        cs.assert_called('POST', '/volumes/1234/action')

    def test_roll_detaching(self):
        v = cs.volumes.get('1234')
        self._assert_request_id(v)
        vol = cs.volumes.roll_detaching(v)
        cs.assert_called('POST', '/volumes/1234/action')
        self._assert_request_id(vol)

    def test_initialize_connection(self):
        v = cs.volumes.get('1234')
        self._assert_request_id(v)
        vol = cs.volumes.initialize_connection(v, {})
        cs.assert_called('POST', '/volumes/1234/action')
        self._assert_request_id(vol)

    def test_terminate_connection(self):
        v = cs.volumes.get('1234')
        self._assert_request_id(v)
        vol = cs.volumes.terminate_connection(v, {})
        cs.assert_called('POST', '/volumes/1234/action')
        self._assert_request_id(vol)

    def test_set_metadata(self):
        vol = cs.volumes.set_metadata(1234, {'k1': 'v2', 'тест': 'тест'})
        cs.assert_called('POST', '/volumes/1234/metadata', {'metadata': {'k1': 'v2', 'тест': 'тест'}})
        self._assert_request_id(vol)

    def test_delete_metadata(self):
        keys = ['key1']
        vol = cs.volumes.delete_metadata(1234, keys)
        cs.assert_called('DELETE', '/volumes/1234/metadata/key1')
        self._assert_request_id(vol)

    def test_extend(self):
        v = cs.volumes.get('1234')
        self._assert_request_id(v)
        vol = cs.volumes.extend(v, 2)
        cs.assert_called('POST', '/volumes/1234/action')
        self._assert_request_id(vol)

    def test_reset_state(self):
        v = cs.volumes.get('1234')
        self._assert_request_id(v)
        vol = cs.volumes.reset_state(v, 'in-use', attach_status='detached')
        cs.assert_called('POST', '/volumes/1234/action')
        self._assert_request_id(vol)

    def test_reset_state_migration_status(self):
        v = cs.volumes.get('1234')
        self._assert_request_id(v)
        vol = cs.volumes.reset_state(v, 'in-use', attach_status='detached', migration_status='none')
        cs.assert_called('POST', '/volumes/1234/action')
        self._assert_request_id(vol)

    def test_get_encryption_metadata(self):
        vol = cs.volumes.get_encryption_metadata('1234')
        cs.assert_called('GET', '/volumes/1234/encryption')
        self._assert_request_id(vol)

    def test_migrate(self):
        v = cs.volumes.get('1234')
        self._assert_request_id(v)
        vol = cs.volumes.migrate_volume(v, 'dest', False, False)
        cs.assert_called('POST', '/volumes/1234/action', {'os-migrate_volume': {'host': 'dest', 'force_host_copy': False, 'lock_volume': False}})
        self._assert_request_id(vol)

    def test_migrate_with_lock_volume(self):
        v = cs.volumes.get('1234')
        self._assert_request_id(v)
        vol = cs.volumes.migrate_volume(v, 'dest', False, True)
        cs.assert_called('POST', '/volumes/1234/action', {'os-migrate_volume': {'host': 'dest', 'force_host_copy': False, 'lock_volume': True}})
        self._assert_request_id(vol)

    def test_metadata_update_all(self):
        vol = cs.volumes.update_all_metadata(1234, {'k1': 'v1'})
        cs.assert_called('PUT', '/volumes/1234/metadata', {'metadata': {'k1': 'v1'}})
        self._assert_request_id(vol)

    def test_readonly_mode_update(self):
        v = cs.volumes.get('1234')
        self._assert_request_id(v)
        vol = cs.volumes.update_readonly_flag(v, True)
        cs.assert_called('POST', '/volumes/1234/action')
        self._assert_request_id(vol)

    def test_retype(self):
        v = cs.volumes.get('1234')
        self._assert_request_id(v)
        vol = cs.volumes.retype(v, 'foo', 'on-demand')
        cs.assert_called('POST', '/volumes/1234/action', {'os-retype': {'new_type': 'foo', 'migration_policy': 'on-demand'}})
        self._assert_request_id(vol)

    def test_set_bootable(self):
        v = cs.volumes.get('1234')
        self._assert_request_id(v)
        vol = cs.volumes.set_bootable(v, True)
        cs.assert_called('POST', '/volumes/1234/action')
        self._assert_request_id(vol)

    def test_volume_manage(self):
        vol = cs.volumes.manage('host1', {'k': 'v'})
        expected = {'host': 'host1', 'name': None, 'availability_zone': None, 'description': None, 'metadata': None, 'ref': {'k': 'v'}, 'volume_type': None, 'bootable': False}
        cs.assert_called('POST', '/os-volume-manage', {'volume': expected})
        self._assert_request_id(vol)

    def test_volume_manage_bootable(self):
        vol = cs.volumes.manage('host1', {'k': 'v'}, bootable=True)
        expected = {'host': 'host1', 'name': None, 'availability_zone': None, 'description': None, 'metadata': None, 'ref': {'k': 'v'}, 'volume_type': None, 'bootable': True}
        cs.assert_called('POST', '/os-volume-manage', {'volume': expected})
        self._assert_request_id(vol)

    def test_volume_list_manageable(self):
        cs.volumes.list_manageable('host1', detailed=False)
        cs.assert_called('GET', '/os-volume-manage?host=host1')

    def test_volume_list_manageable_detailed(self):
        cs.volumes.list_manageable('host1', detailed=True)
        cs.assert_called('GET', '/os-volume-manage/detail?host=host1')

    def test_volume_unmanage(self):
        v = cs.volumes.get('1234')
        self._assert_request_id(v)
        vol = cs.volumes.unmanage(v)
        cs.assert_called('POST', '/volumes/1234/action', {'os-unmanage': None})
        self._assert_request_id(vol)

    def test_snapshot_manage(self):
        vol = cs.volume_snapshots.manage('volume_id1', {'k': 'v'})
        expected = {'volume_id': 'volume_id1', 'name': None, 'description': None, 'metadata': None, 'ref': {'k': 'v'}}
        cs.assert_called('POST', '/os-snapshot-manage', {'snapshot': expected})
        self._assert_request_id(vol)

    def test_snapshot_list_manageable(self):
        cs.volume_snapshots.list_manageable('host1', detailed=False)
        cs.assert_called('GET', '/os-snapshot-manage?host=host1')

    def test_snapshot_list_manageable_detailed(self):
        cs.volume_snapshots.list_manageable('host1', detailed=True)
        cs.assert_called('GET', '/os-snapshot-manage/detail?host=host1')

    def test_get_pools(self):
        vol = cs.volumes.get_pools('')
        cs.assert_called('GET', '/scheduler-stats/get_pools')
        self._assert_request_id(vol)

    def test_get_pools_detail(self):
        vol = cs.volumes.get_pools('--detail')
        cs.assert_called('GET', '/scheduler-stats/get_pools?detail=True')
        self._assert_request_id(vol)