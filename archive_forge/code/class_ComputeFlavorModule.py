from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class ComputeFlavorModule(OpenStackModule):
    argument_spec = dict(description=dict(), disk=dict(type='int'), ephemeral=dict(type='int'), extra_specs=dict(type='dict'), id=dict(aliases=['flavorid']), is_public=dict(type='bool'), name=dict(required=True), ram=dict(type='int'), rxtx_factor=dict(type='float'), state=dict(default='present', choices=['absent', 'present']), swap=dict(type='int'), vcpus=dict(type='int'))
    module_kwargs = dict(required_if=[('state', 'present', ['ram', 'vcpus', 'disk'])], supports_check_mode=True)

    def run(self):
        state = self.params['state']
        id = self.params['id']
        name = self.params['name']
        name_or_id = id if id and id != 'auto' else name
        flavor = self.conn.compute.find_flavor(name_or_id, get_extra_specs=True)
        if self.ansible.check_mode:
            self.exit_json(changed=self._will_change(state, flavor))
        if state == 'present' and (not flavor):
            flavor = self._create()
            self.exit_json(changed=True, flavor=flavor.to_dict(computed=False))
        elif state == 'present' and flavor:
            update = self._build_update(flavor)
            if update:
                flavor = self._update(flavor, update)
            self.exit_json(changed=bool(update), flavor=flavor.to_dict(computed=False))
        elif state == 'absent' and flavor:
            self._delete(flavor)
            self.exit_json(changed=True)
        elif state == 'absent' and (not flavor):
            self.exit_json(changed=False)

    def _build_update(self, flavor):
        return {**self._build_update_extra_specs(flavor), **self._build_update_flavor(flavor)}

    def _build_update_extra_specs(self, flavor):
        update = {}
        old_extra_specs = flavor['extra_specs']
        new_extra_specs = self.params['extra_specs'] or {}
        if flavor['swap'] == '':
            flavor['swap'] = 0
        delete_extra_specs_keys = set(old_extra_specs.keys()) - set(new_extra_specs.keys())
        if delete_extra_specs_keys:
            update['delete_extra_specs_keys'] = delete_extra_specs_keys
        stringified = dict([(k, str(v)) for k, v in new_extra_specs.items()])
        if old_extra_specs != stringified:
            update['create_extra_specs'] = new_extra_specs
        return update

    def _build_update_flavor(self, flavor):
        update = {}
        flavor_attributes = dict(((k, self.params[k]) for k in ['ram', 'vcpus', 'disk', 'ephemeral', 'swap', 'rxtx_factor', 'is_public', 'description'] if k in self.params and self.params[k] is not None and (self.params[k] != flavor[k])))
        if flavor_attributes:
            update['flavor_attributes'] = flavor_attributes
        return update

    def _create(self):
        kwargs = dict(((k, self.params[k]) for k in ['name', 'ram', 'vcpus', 'disk', 'ephemeral', 'swap', 'rxtx_factor', 'is_public', 'description'] if self.params[k] is not None))
        id = self.params['id']
        if id is not None and id != 'auto':
            kwargs['id'] = id
        flavor = self.conn.compute.create_flavor(**kwargs)
        extra_specs = self.params['extra_specs']
        if extra_specs:
            flavor = self.conn.compute.create_flavor_extra_specs(flavor.id, extra_specs)
        return flavor

    def _delete(self, flavor):
        self.conn.compute.delete_flavor(flavor)

    def _update(self, flavor, update):
        flavor = self._update_flavor(flavor, update)
        flavor = self._update_extra_specs(flavor, update)
        return flavor

    def _update_extra_specs(self, flavor, update):
        if update.get('flavor_attributes'):
            return flavor
        delete_extra_specs_keys = update.get('delete_extra_specs_keys')
        if delete_extra_specs_keys:
            self.conn.unset_flavor_specs(flavor.id, delete_extra_specs_keys)
            flavor = self.conn.compute.fetch_flavor_extra_specs(flavor)
        create_extra_specs = update.get('create_extra_specs')
        if create_extra_specs:
            flavor = self.conn.compute.create_flavor_extra_specs(flavor.id, create_extra_specs)
        return flavor

    def _update_flavor(self, flavor, update):
        flavor_attributes = update.get('flavor_attributes')
        if flavor_attributes:
            self._delete(flavor)
            flavor = self._create()
        return flavor

    def _will_change(self, state, flavor):
        if state == 'present' and (not flavor):
            return True
        elif state == 'present' and flavor:
            return bool(self._build_update(flavor))
        elif state == 'absent' and flavor:
            return True
        else:
            return False