from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class IdentityFederationMappingModule(OpenStackModule):
    argument_spec = dict(name=dict(required=True, aliases=['id']), rules=dict(type='list', elements='dict', options=dict(local=dict(required=True, type='list', elements='dict'), remote=dict(required=True, type='list', elements='dict'))), state=dict(default='present', choices=['absent', 'present']))
    module_kwargs = dict(required_if=[('state', 'present', ['rules'])], supports_check_mode=True)

    def run(self):
        state = self.params['state']
        id = self.params['name']
        mapping = self.conn.identity.find_mapping(id)
        if self.ansible.check_mode:
            self.exit_json(changed=self._will_change(state, mapping))
        if state == 'present' and (not mapping):
            mapping = self._create()
            self.exit_json(changed=True, mapping=mapping.to_dict(computed=False))
        elif state == 'present' and mapping:
            update = self._build_update(mapping)
            if update:
                mapping = self._update(mapping, update)
            self.exit_json(changed=bool(update), mapping=mapping.to_dict(computed=False))
        elif state == 'absent' and mapping:
            self._delete(mapping)
            self.exit_json(changed=True)
        elif state == 'absent' and (not mapping):
            self.exit_json(changed=False)

    def _build_update(self, mapping):
        update = {}
        if len(self.params['rules']) < 1:
            self.fail_json(msg='At least one rule must be passed')
        attributes = dict(((k, self.params[k]) for k in ['rules'] if k in self.params and self.params[k] is not None and (self.params[k] != mapping[k])))
        if attributes:
            update['attributes'] = attributes
        return update

    def _create(self):
        return self.conn.identity.create_mapping(id=self.params['name'], rules=self.params['rules'])

    def _delete(self, mapping):
        self.conn.identity.delete_mapping(mapping.id)

    def _update(self, mapping, update):
        attributes = update.get('attributes')
        if attributes:
            mapping = self.conn.identity.update_mapping(mapping.id, **attributes)
        return mapping

    def _will_change(self, state, mapping):
        if state == 'present' and (not mapping):
            return True
        elif state == 'present' and mapping:
            return bool(self._build_update(mapping))
        elif state == 'absent' and mapping:
            return True
        else:
            return False