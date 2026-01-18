from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class IdentityEndpointModule(OpenStackModule):
    argument_spec = dict(service=dict(required=True), endpoint_interface=dict(required=True, choices=['admin', 'public', 'internal']), url=dict(required=True), region=dict(), enabled=dict(type='bool', default=True), state=dict(default='present', choices=['absent', 'present']))
    module_kwargs = dict(supports_check_mode=True)

    def _needs_update(self, endpoint):
        if endpoint.is_enabled != self.params['enabled']:
            return True
        if endpoint.url != self.params['url']:
            return True
        return False

    def _system_state_change(self, endpoint):
        state = self.params['state']
        if state == 'absent' and endpoint:
            return True
        if state == 'present':
            if endpoint is None:
                return True
            return self._needs_update(endpoint)
        return False

    def run(self):
        service_name_or_id = self.params['service']
        interface = self.params['endpoint_interface']
        url = self.params['url']
        region_id = self.params['region']
        enabled = self.params['enabled']
        state = self.params['state']
        service = self.conn.identity.find_service(service_name_or_id)
        if service is None and state == 'absent':
            self.exit_json(changed=False)
        if service is None and state == 'present':
            self.fail_json(msg='Service %s does not exist' % service_name_or_id)
        filters = dict(service_id=service.id, interface=interface)
        if region_id:
            filters['region_id'] = region_id
        endpoints = list(self.conn.identity.endpoints(**filters))
        endpoint = None
        if len(endpoints) > 1:
            self.fail_json(msg='Service %s, interface %s and region %s are not unique' % (service_name_or_id, interface, region_id))
        elif len(endpoints) == 1:
            endpoint = endpoints[0]
        if self.ansible.check_mode:
            self.exit_json(changed=self._system_state_change(endpoint))
        changed = False
        if state == 'present':
            if not endpoint:
                args = {'url': url, 'interface': interface, 'service_id': service.id, 'enabled': enabled, 'region_id': region_id}
                endpoint = self.conn.identity.create_endpoint(**args)
                changed = True
            elif self._needs_update(endpoint):
                endpoint = self.conn.identity.update_endpoint(endpoint.id, url=url, enabled=enabled)
                changed = True
            self.exit_json(changed=changed, endpoint=endpoint.to_dict(computed=False))
        elif state == 'absent' and endpoint:
            self.conn.identity.delete_endpoint(endpoint.id)
            changed = True
        self.exit_json(changed=changed)