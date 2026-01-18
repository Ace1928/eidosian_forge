from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class StackModule(OpenStackModule):
    argument_spec = dict(environment=dict(type='list', elements='str'), name=dict(required=True), parameters=dict(default={}, type='dict'), rollback=dict(default=False, type='bool'), state=dict(default='present', choices=['absent', 'present']), tags=dict(aliases=['tag']), template=dict(), timeout=dict(default=3600, type='int'))
    module_kwargs = dict(supports_check_mode=True, required_if=[('state', 'present', ('template',), True)])

    def _system_state_change(self, stack, state):
        if state == 'present':
            return True
        if state == 'absent' and stack:
            return True
        return False

    def run(self):
        state = self.params['state']
        name = self.params['name']
        stack = self.conn.get_stack(name)
        if self.ansible.check_mode:
            self.exit_json(changed=self._system_state_change(stack, state))
        if state == 'present':
            is_update = bool(stack)
            kwargs = dict(template_file=self.params['template'], environment_files=self.params['environment'], timeout=self.params['timeout'], rollback=self.params['rollback'], wait=True)
            tags = self.params['tags']
            if tags is not None:
                kwargs['tags'] = tags
            extra_kwargs = self.params['parameters']
            dup_kwargs = set(kwargs.keys()) & set(extra_kwargs.keys())
            if dup_kwargs:
                raise ValueError('Duplicate key(s) {0} in parameters'.format(list(dup_kwargs)))
            kwargs = dict(kwargs, **extra_kwargs)
            if not is_update:
                stack = self.conn.create_stack(name, **kwargs)
            else:
                stack = self.conn.update_stack(name, **kwargs)
            stack = self.conn.orchestration.get_stack(stack['id'])
            self.exit_json(changed=True, stack=stack.to_dict(computed=False))
        elif state == 'absent':
            if not stack:
                self.exit_json(changed=False)
            else:
                self.conn.delete_stack(name_or_id=stack['id'], wait=self.params['wait'])
                self.exit_json(changed=True)