from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class VolumeSnapshotInfoModule(OpenStackModule):
    argument_spec = dict(details=dict(type='bool'), name=dict(), status=dict(choices=['available', 'backing-up', 'creating', 'deleted', 'deleting', 'error', 'error_deleting', 'restoring', 'unmanaging']), volume=dict())
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        kwargs = dict(((k, self.params[k]) for k in ['details', 'name', 'status'] if self.params[k] is not None))
        volume_name_or_id = self.params['volume']
        volume = None
        if volume_name_or_id:
            volume = self.conn.block_storage.find_volume(volume_name_or_id)
            if volume:
                kwargs['volume_id'] = volume.id
        if volume_name_or_id and (not volume):
            snapshots = []
        else:
            snapshots = [b.to_dict(computed=False) for b in self.conn.block_storage.snapshots(**kwargs)]
        self.exit_json(changed=False, volume_snapshots=snapshots)