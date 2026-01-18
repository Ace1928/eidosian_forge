from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class ServerVolumeModule(OpenStackModule):
    argument_spec = dict(server=dict(required=True), volume=dict(required=True), device=dict(), state=dict(default='present', choices=['absent', 'present']))

    def run(self):
        state = self.params['state']
        wait = self.params['wait']
        timeout = self.params['timeout']
        server = self.conn.compute.find_server(self.params['server'], ignore_missing=False)
        volume = self.conn.block_storage.find_volume(self.params['volume'], ignore_missing=False)
        dev = self.conn.get_volume_attach_device(volume, server.id)
        if self.ansible.check_mode:
            self.exit_json(changed=_system_state_change(state, dev))
        if state == 'present':
            changed = False
            if not dev:
                changed = True
                self.conn.attach_volume(server, volume, device=self.params['device'], wait=wait, timeout=timeout)
                volume = self.conn.block_storage.get_volume(volume.id)
            self.exit_json(changed=changed, volume=volume.to_dict(computed=False))
        elif state == 'absent':
            if not dev:
                self.exit_json(changed=False)
            self.conn.detach_volume(server, volume, wait=wait, timeout=timeout)
            self.exit_json(changed=True)