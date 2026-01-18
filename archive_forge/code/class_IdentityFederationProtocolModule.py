from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class IdentityFederationProtocolModule(OpenStackModule):
    argument_spec = dict(name=dict(required=True, aliases=['id']), state=dict(default='present', choices=['absent', 'present']), idp=dict(required=True, aliases=['idp_id', 'idp_name']), mapping=dict(aliases=['mapping_id', 'mapping_name']))
    module_kwargs = dict(required_if=[('state', 'present', ('mapping',))], supports_check_mode=True)

    def run(self):
        state = self.params['state']
        id = self.params['name']
        idp_id = self.params['idp']
        protocol = self.conn.identity.find_federation_protocol(idp_id, id)
        if self.ansible.check_mode:
            self.exit_json(changed=self._will_change(state, protocol))
        if state == 'present' and (not protocol):
            protocol = self._create()
            self.exit_json(changed=True, protocol=protocol.to_dict(computed=False))
        elif state == 'present' and protocol:
            update = self._build_update(protocol)
            if update:
                protocol = self._update(protocol, update)
            self.exit_json(changed=bool(update), protocol=protocol.to_dict(computed=False))
        elif state == 'absent' and protocol:
            self._delete(protocol)
            self.exit_json(changed=True)
        elif state == 'absent' and (not protocol):
            self.exit_json(changed=False)

    def _build_update(self, protocol):
        update = {}
        attributes = dict(((k, self.params[p]) for p, k in {'mapping': 'mapping_id'}.items() if p in self.params and self.params[p] is not None and (self.params[p] != protocol[k])))
        if attributes:
            update['attributes'] = attributes
        return update

    def _create(self):
        return self.conn.identity.create_federation_protocol(id=self.params['name'], idp_id=self.params['idp'], mapping_id=self.params['mapping'])

    def _delete(self, protocol):
        self.conn.identity.delete_federation_protocol(None, protocol)

    def _update(self, protocol, update):
        attributes = update.get('attributes')
        if attributes:
            protocol = self.conn.identity.update_federation_protocol(protocol.idp_id, protocol.id, **attributes)
        return protocol

    def _will_change(self, state, protocol):
        if state == 'present' and (not protocol):
            return True
        elif state == 'present' and protocol:
            return bool(self._build_update(protocol))
        elif state == 'absent' and protocol:
            return True
        else:
            return False