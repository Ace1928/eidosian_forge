from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class VolumeBackupModule(OpenStackModule):
    argument_spec = dict(description=dict(aliases=['display_description']), force=dict(default=False, type='bool'), is_incremental=dict(default=False, type='bool', aliases=['incremental']), metadata=dict(type='dict'), name=dict(required=True, aliases=['display_name']), snapshot=dict(), state=dict(default='present', choices=['absent', 'present']), volume=dict())
    module_kwargs = dict(required_if=[('state', 'present', ['volume'])], supports_check_mode=True)

    def run(self):
        name = self.params['name']
        state = self.params['state']
        backup = self.conn.block_storage.find_backup(name)
        if self.ansible.check_mode:
            self.exit_json(changed=self._will_change(state, backup))
        if state == 'present' and (not backup):
            backup = self._create()
            self.exit_json(changed=True, backup=backup.to_dict(computed=False), volume_backup=backup.to_dict(computed=False))
        elif state == 'present' and backup:
            self.exit_json(changed=False, backup=backup.to_dict(computed=False), volume_backup=backup.to_dict(computed=False))
        elif state == 'absent' and backup:
            self._delete(backup)
            self.exit_json(changed=True)
        else:
            self.exit_json(changed=False)

    def _create(self):
        args = dict()
        for k in ['description', 'is_incremental', 'force', 'metadata', 'name']:
            if self.params[k] is not None:
                args[k] = self.params[k]
        volume_name_or_id = self.params['volume']
        volume = self.conn.block_storage.find_volume(volume_name_or_id, ignore_missing=False)
        args['volume_id'] = volume.id
        snapshot_name_or_id = self.params['snapshot']
        if snapshot_name_or_id:
            snapshot = self.conn.block_storage.find_snapshot(snapshot_name_or_id, ignore_missing=False)
            args['snapshot_id'] = snapshot.id
        backup = self.conn.block_storage.create_backup(**args)
        if self.params['wait']:
            backup = self.conn.block_storage.wait_for_status(backup, status='available', wait=self.params['timeout'])
        return backup

    def _delete(self, backup):
        self.conn.block_storage.delete_backup(backup)
        if self.params['wait']:
            self.conn.block_storage.wait_for_delete(backup, wait=self.params['timeout'])

    def _will_change(self, state, backup):
        if state == 'present' and (not backup):
            return True
        elif state == 'present' and backup:
            return False
        elif state == 'absent' and backup:
            return True
        else:
            return False