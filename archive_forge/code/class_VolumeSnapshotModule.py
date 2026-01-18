from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class VolumeSnapshotModule(OpenStackModule):
    argument_spec = dict(description=dict(aliases=['display_description']), name=dict(required=True, aliases=['display_name']), force=dict(default=False, type='bool'), state=dict(default='present', choices=['absent', 'present']), volume=dict())
    module_kwargs = dict(required_if=[('state', 'present', ['volume'])], supports_check_mode=True)

    def run(self):
        name = self.params['name']
        state = self.params['state']
        snapshot = self.conn.block_storage.find_snapshot(name)
        if self.ansible.check_mode:
            self.exit_json(changed=self._will_change(state, snapshot))
        if state == 'present' and (not snapshot):
            snapshot = self._create()
            self.exit_json(changed=True, snapshot=snapshot.to_dict(computed=False), volume_snapshot=snapshot.to_dict(computed=False))
        elif state == 'present' and snapshot:
            self.exit_json(changed=False, snapshot=snapshot.to_dict(computed=False), volume_snapshot=snapshot.to_dict(computed=False))
        elif state == 'absent' and snapshot:
            self._delete(snapshot)
            self.exit_json(changed=True)
        else:
            self.exit_json(changed=False)

    def _create(self):
        args = dict()
        for k in ['description', 'force', 'name']:
            if self.params[k] is not None:
                args[k] = self.params[k]
        volume_name_or_id = self.params['volume']
        volume = self.conn.block_storage.find_volume(volume_name_or_id, ignore_missing=False)
        args['volume_id'] = volume.id
        snapshot = self.conn.block_storage.create_snapshot(**args)
        if self.params['wait']:
            snapshot = self.conn.block_storage.wait_for_status(snapshot, wait=self.params['timeout'])
        return snapshot

    def _delete(self, snapshot):
        self.conn.block_storage.delete_snapshot(snapshot)
        if self.params['wait']:
            self.conn.block_storage.wait_for_delete(snapshot, wait=self.params['timeout'])

    def _will_change(self, state, snapshot):
        if state == 'present' and (not snapshot):
            return True
        elif state == 'present' and snapshot:
            return False
        elif state == 'absent' and snapshot:
            return True
        else:
            return False