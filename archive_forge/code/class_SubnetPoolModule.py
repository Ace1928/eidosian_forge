from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class SubnetPoolModule(OpenStackModule):
    argument_spec = dict(address_scope=dict(), default_prefix_length=dict(type='int'), default_quota=dict(type='int'), description=dict(), extra_specs=dict(type='dict'), is_default=dict(type='bool'), is_shared=dict(type='bool', aliases=['shared']), maximum_prefix_length=dict(type='int'), minimum_prefix_length=dict(type='int'), name=dict(required=True), prefixes=dict(type='list', elements='str'), project=dict(), state=dict(default='present', choices=['absent', 'present']))

    def run(self):
        state = self.params['state']
        name = self.params['name']
        subnet_pool = self.conn.network.find_subnet_pool(name)
        if self.ansible.check_mode:
            self.exit_json(changed=self._will_change(state, subnet_pool))
        if state == 'present' and (not subnet_pool):
            subnet_pool = self._create()
            self.exit_json(changed=True, subnet_pool=subnet_pool.to_dict(computed=False))
        elif state == 'present' and subnet_pool:
            update = self._build_update(subnet_pool)
            if update:
                subnet_pool = self._update(subnet_pool, update)
            self.exit_json(changed=bool(update), subnet_pool=subnet_pool.to_dict(computed=False))
        elif state == 'absent' and subnet_pool:
            self._delete(subnet_pool)
            self.exit_json(changed=True)
        elif state == 'absent' and (not subnet_pool):
            self.exit_json(changed=False)

    def _build_update(self, subnet_pool):
        update = {}
        attributes = dict(((k, self.params[k]) for k in ['default_prefix_length', 'default_quota', 'description', 'is_default', 'maximum_prefix_length', 'minimum_prefix_length'] if self.params[k] is not None and self.params[k] != subnet_pool[k]))
        for k in ['prefixes']:
            if self.params[k] is not None and set(self.params[k]) != set(subnet_pool[k]):
                attributes[k] = self.params[k]
        project_name_or_id = self.params['project']
        if project_name_or_id is not None:
            project = self.conn.identity.find_project(project_name_or_id, ignore_missing=False)
            if subnet_pool['project_id'] != project.id:
                attributes['project_id'] = project.id
        address_scope_name_or_id = self.params['address_scope']
        if address_scope_name_or_id is not None:
            address_scope = self.conn.network.find_address_scope(address_scope_name_or_id, ignore_missing=False)
            if subnet_pool['address_scope_id'] != address_scope.id:
                attributes['address_scope_id'] = address_scope.id
        extra_specs = self.params['extra_specs']
        if extra_specs:
            duplicate_keys = set(attributes.keys()) & set(extra_specs.keys())
            if duplicate_keys:
                raise ValueError('Duplicate key(s) in extra_specs: {0}'.format(', '.join(list(duplicate_keys))))
            for k, v in extra_specs.items():
                if v != subnet_pool[k]:
                    attributes[k] = v
        if attributes:
            update['attributes'] = attributes
        return update

    def _create(self):
        kwargs = dict(((k, self.params[k]) for k in ['default_prefix_length', 'default_quota', 'description', 'is_default', 'is_shared', 'maximum_prefix_length', 'minimum_prefix_length', 'name', 'prefixes'] if self.params[k] is not None))
        project_name_or_id = self.params['project']
        if project_name_or_id is not None:
            project = self.conn.identity.find_project(project_name_or_id, ignore_missing=False)
            kwargs['project_id'] = project.id
        address_scope_name_or_id = self.params['address_scope']
        if address_scope_name_or_id is not None:
            address_scope = self.conn.network.find_address_scope(address_scope_name_or_id, ignore_missing=False)
            kwargs['address_scope_id'] = address_scope.id
        extra_specs = self.params['extra_specs']
        if extra_specs:
            duplicate_keys = set(kwargs.keys()) & set(extra_specs.keys())
            if duplicate_keys:
                raise ValueError('Duplicate key(s) in extra_specs: {0}'.format(', '.join(list(duplicate_keys))))
            kwargs = dict(kwargs, **extra_specs)
        return self.conn.network.create_subnet_pool(**kwargs)

    def _delete(self, subnet_pool):
        self.conn.network.delete_subnet_pool(subnet_pool.id)

    def _update(self, subnet_pool, update):
        attributes = update.get('attributes')
        if attributes:
            subnet_pool = self.conn.network.update_subnet_pool(subnet_pool.id, **attributes)
        return subnet_pool

    def _will_change(self, state, subnet_pool):
        if state == 'present' and (not subnet_pool):
            return True
        elif state == 'present' and subnet_pool:
            return bool(self._build_update(subnet_pool))
        elif state == 'absent' and subnet_pool:
            return True
        else:
            return False