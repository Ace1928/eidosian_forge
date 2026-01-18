from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class LoadBalancerListenerModule(OpenStackModule):
    argument_spec = dict(default_tls_container_ref=dict(), description=dict(), is_admin_state_up=dict(type='bool'), load_balancer=dict(aliases=['loadbalancer']), name=dict(required=True), protocol=dict(default='HTTP'), protocol_port=dict(type='int'), sni_container_refs=dict(type='list', elements='str'), state=dict(default='present', choices=['absent', 'present']), timeout_client_data=dict(type='int'), timeout_member_data=dict(type='int'))
    module_kwargs = dict(required_if=[('state', 'present', ('load_balancer',))], supports_check_mode=True)

    def run(self):
        state = self.params['state']
        listener = self._find()
        if self.ansible.check_mode:
            self.exit_json(changed=self._will_change(state, listener))
        if state == 'present' and (not listener):
            listener = self._create()
            self.exit_json(changed=True, rbac_listener=listener.to_dict(computed=False), listener=listener.to_dict(computed=False))
        elif state == 'present' and listener:
            update = self._build_update(listener)
            if update:
                listener = self._update(listener, update)
            self.exit_json(changed=bool(update), rbac_listener=listener.to_dict(computed=False), listener=listener.to_dict(computed=False))
        elif state == 'absent' and listener:
            self._delete(listener)
            self.exit_json(changed=True)
        elif state == 'absent' and (not listener):
            self.exit_json(changed=False)

    def _build_update(self, listener):
        update = {}
        non_updateable_keys = [k for k in ['protocol', 'protocol_port'] if self.params[k] is not None and self.params[k] != listener[k]]
        load_balancer_name_or_id = self.params['load_balancer']
        load_balancer = self.conn.load_balancer.find_load_balancer(load_balancer_name_or_id, ignore_missing=False)
        if listener['load_balancers'] != [dict(id=load_balancer.id)]:
            non_updateable_keys.append('load_balancer')
        if non_updateable_keys:
            self.fail_json(msg='Cannot update parameters {0}'.format(non_updateable_keys))
        attributes = dict(((k, self.params[k]) for k in ['default_tls_container_ref', 'description', 'is_admin_state_up', 'sni_container_refs', 'timeout_client_data', 'timeout_member_data'] if self.params[k] is not None and self.params[k] != listener[k]))
        if attributes:
            update['attributes'] = attributes
        return update

    def _create(self):
        kwargs = dict(((k, self.params[k]) for k in ['default_tls_container_ref', 'description', 'is_admin_state_up', 'name', 'protocol', 'protocol_port', 'sni_container_refs', 'timeout_client_data', 'timeout_member_data'] if self.params[k] is not None))
        load_balancer_name_or_id = self.params['load_balancer']
        load_balancer = self.conn.load_balancer.find_load_balancer(load_balancer_name_or_id, ignore_missing=False)
        kwargs['load_balancer_id'] = load_balancer.id
        listener = self.conn.load_balancer.create_listener(**kwargs)
        if self.params['wait']:
            self.conn.load_balancer.wait_for_load_balancer(listener.load_balancer_id, wait=self.params['timeout'])
        return listener

    def _delete(self, listener):
        self.conn.load_balancer.delete_listener(listener.id)
        if self.params['wait']:
            if not listener.load_balancers or len(listener.load_balancers) != 1:
                raise AssertionError('A single load-balancer is expected')
            self.conn.load_balancer.wait_for_load_balancer(listener.load_balancers[0]['id'], wait=self.params['timeout'])

    def _find(self):
        name = self.params['name']
        return self.conn.load_balancer.find_listener(name_or_id=name)

    def _update(self, listener, update):
        attributes = update.get('attributes')
        if attributes:
            listener = self.conn.load_balancer.update_listener(listener.id, **attributes)
        if self.params['wait']:
            if not listener.load_balancers or len(listener.load_balancers) != 1:
                raise AssertionError('A single load-balancer is expected')
            self.conn.load_balancer.wait_for_load_balancer(listener.load_balancers[0]['id'], wait=self.params['timeout'])
        return listener

    def _will_change(self, state, listener):
        if state == 'present' and (not listener):
            return True
        elif state == 'present' and listener:
            return bool(self._build_update(listener))
        elif state == 'absent' and listener:
            return True
        else:
            return False