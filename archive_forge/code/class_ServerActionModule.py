from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class ServerActionModule(OpenStackModule):
    argument_spec = dict(action=dict(required=True, choices=['stop', 'start', 'pause', 'unpause', 'lock', 'unlock', 'suspend', 'reboot_soft', 'reboot_hard', 'resume', 'rebuild', 'shelve', 'shelve_offload', 'unshelve']), admin_password=dict(no_log=True), all_projects=dict(type='bool', default=False), image=dict(), name=dict(required=True, aliases=['server']))
    module_kwargs = dict(required_if=[('action', 'rebuild', ['image'])], supports_check_mode=True)
    _action_map = {'stop': ['SHUTOFF'], 'start': ['ACTIVE'], 'pause': ['PAUSED'], 'unpause': ['ACTIVE'], 'lock': ['ACTIVE'], 'unlock': ['ACTIVE'], 'suspend': ['SUSPENDED'], 'reboot_soft': ['ACTIVE'], 'reboot_hard': ['ACTIVE'], 'resume': ['ACTIVE'], 'rebuild': ['ACTIVE'], 'shelve': ['SHELVED_OFFLOADED', 'SHELVED'], 'shelve_offload': ['SHELVED_OFFLOADED'], 'unshelve': ['ACTIVE']}

    def run(self):
        server = self.conn.get_server(name_or_id=self.params['name'], detailed=True, all_projects=self.params['all_projects'])
        if not server:
            self.fail_json(msg='No Server found for {0}'.format(self.params['name']))
        action = self.params['action']
        will_change = action == 'rebuild' or (action == 'lock' and (not server['is_locked'])) or (action == 'unlock' and server['is_locked']) or (server.status.lower() not in [a.lower() for a in self._action_map[action]])
        if not will_change:
            self.exit_json(changed=False)
        elif self.ansible.check_mode:
            self.exit_json(changed=True)
        if action == 'rebuild':
            image = self.conn.image.find_image(self.params['image'], ignore_missing=False)
            kwargs = dict(server=server, name=server['name'], image=image['id'])
            admin_password = self.params['admin_password']
            if admin_password is not None:
                kwargs['admin_password'] = admin_password
            self.conn.compute.rebuild_server(**kwargs)
        elif action == 'shelve_offload':
            response = self.conn.compute.post('/servers/{server_id}/action'.format(server_id=server['id']), json={'shelveOffload': None})
            self.sdk.exceptions.raise_from_response(response)
        else:
            action_name = action + '_server'
            if action in ['reboot_soft', 'reboot_hard']:
                action_name = 'reboot_server'
            func_name = getattr(self.conn.compute, action_name)
            if action == 'reboot_soft':
                func_name(server, 'SOFT')
            elif action == 'reboot_hard':
                func_name(server, 'HARD')
            else:
                func_name(server)
        if self.params['wait']:
            for count in self.sdk.utils.iterate_timeout(timeout=self.params['timeout'], message='Timeout waiting for action {0} to be completed.'.format(action)):
                server = self.conn.compute.get_server(server['id'])
                if action == 'lock' and server['is_locked'] or (action == 'unlock' and (not server['is_locked'])):
                    break
                states = [s.lower() for s in self._action_map[action]]
                if server.status.lower() in states:
                    break
        self.exit_json(changed=True)