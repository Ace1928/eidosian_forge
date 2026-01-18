from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class VolumeModule(OpenStackModule):
    argument_spec = dict(availability_zone=dict(), description=dict(aliases=['display_description']), image=dict(), is_bootable=dict(type='bool', default=False, aliases=['bootable']), is_multiattach=dict(type='bool'), metadata=dict(type='dict'), name=dict(required=True, aliases=['display_name']), scheduler_hints=dict(type='dict'), size=dict(type='int'), snapshot=dict(aliases=['snapshot_id']), state=dict(default='present', choices=['absent', 'present'], type='str'), volume=dict(), volume_type=dict())
    module_kwargs = dict(supports_check_mode=True, mutually_exclusive=[['image', 'snapshot', 'volume']], required_if=[['state', 'present', ['size']]])

    def _build_update(self, volume):
        keys = ('size',)
        return {k: self.params[k] for k in keys if self.params[k] is not None and self.params[k] != volume[k]}

    def _update(self, volume):
        """
        modify volume, the only modification to an existing volume
        available at the moment is extending the size, this is
        limited by the openstacksdk and may change whenever the
        functionality is extended.
        """
        diff = {'before': volume.to_dict(computed=False), 'after': ''}
        diff['after'] = diff['before']
        update = self._build_update(volume)
        if not update:
            self.exit_json(changed=False, volume=volume.to_dict(computed=False), diff=diff)
        if self.ansible.check_mode:
            volume.size = update['size']
            self.exit_json(changed=False, volume=volume.to_dict(computed=False), diff=diff)
        if 'size' in update and update['size'] != volume.size:
            size = update['size']
            self.conn.volume.extend_volume(volume.id, size)
            volume = self.conn.block_storage.get_volume(volume)
        volume = volume.to_dict(computed=False)
        diff['after'] = volume
        self.exit_json(changed=True, volume=volume, diff=diff)

    def _build_create_kwargs(self):
        keys = ('availability_zone', 'is_multiattach', 'size', 'name', 'description', 'volume_type', 'scheduler_hints', 'metadata')
        kwargs = {k: self.params[k] for k in keys if self.params[k] is not None}
        find_filters = {}
        if self.params['snapshot']:
            snapshot = self.conn.block_storage.find_snapshot(self.params['snapshot'], ignore_missing=False, **find_filters)
            kwargs['snapshot_id'] = snapshot.id
        if self.params['image']:
            image = self.conn.image.find_image(self.params['image'], ignore_missing=False)
            kwargs['image_id'] = image.id
        if self.params['volume']:
            volume = self.conn.block_storage.find_volume(self.params['volume'], ignore_missing=False, **find_filters)
            kwargs['source_volume_id'] = volume.id
        return kwargs

    def _create(self):
        diff = {'before': '', 'after': ''}
        volume_args = self._build_create_kwargs()
        if self.ansible.check_mode:
            diff['after'] = volume_args
            self.exit_json(changed=True, volume=volume_args, diff=diff)
        volume = self.conn.block_storage.create_volume(**volume_args)
        if self.params['wait']:
            self.conn.block_storage.wait_for_status(volume, wait=self.params['timeout'])
        volume = volume.to_dict(computed=False)
        diff['after'] = volume
        self.exit_json(changed=True, volume=volume, diff=diff)

    def _delete(self, volume):
        diff = {'before': '', 'after': ''}
        if volume is None:
            self.exit_json(changed=False, diff=diff)
        diff['before'] = volume.to_dict(computed=False)
        if self.ansible.check_mode:
            self.exit_json(changed=True, diff=diff)
        self.conn.block_storage.delete_volume(volume)
        if self.params['wait']:
            self.conn.block_storage.wait_for_delete(volume, wait=self.params['timeout'])
        self.exit_json(changed=True, diff=diff)

    def run(self):
        state = self.params['state']
        volume = self.conn.block_storage.find_volume(self.params['name'])
        if state == 'present':
            if not volume:
                self._create()
            else:
                self._update(volume)
        if state == 'absent':
            self._delete(volume)