from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class DnsZoneModule(OpenStackModule):
    argument_spec = dict(description=dict(), email=dict(), masters=dict(type='list', elements='str'), name=dict(required=True), state=dict(default='present', choices=['absent', 'present']), ttl=dict(type='int'), type=dict(choices=['primary', 'secondary'], aliases=['zone_type']))

    def run(self):
        state = self.params['state']
        name_or_id = self.params['name']
        zone = self.conn.dns.find_zone(name_or_id=name_or_id)
        if self.ansible.check_mode:
            self.exit_json(changed=self._will_change(state, zone))
        if state == 'present' and (not zone):
            zone = self._create()
            self.exit_json(changed=True, zone=zone.to_dict(computed=False))
        elif state == 'present' and zone:
            update = self._build_update(zone)
            if update:
                zone = self._update(zone, update)
            self.exit_json(changed=bool(update), zone=zone.to_dict(computed=False))
        elif state == 'absent' and zone:
            self._delete(zone)
            self.exit_json(changed=True)
        elif state == 'absent' and (not zone):
            self.exit_json(changed=False)

    def _build_update(self, zone):
        update = {}
        attributes = dict(((k, self.params[k]) for k in ['description', 'email', 'masters', 'ttl'] if self.params[k] is not None and self.params[k] != zone[k]))
        if attributes:
            update['attributes'] = attributes
        return update

    def _create(self):
        kwargs = dict(((k, self.params[k]) for k in ['description', 'email', 'masters', 'name', 'ttl', 'type'] if self.params[k] is not None))
        if 'type' in kwargs:
            kwargs['type'] = kwargs['type'].upper()
        zone = self.conn.dns.create_zone(**kwargs)
        if self.params['wait']:
            self.sdk.resource.wait_for_status(self.conn.dns, zone, status='active', failures=['error'], wait=self.params['timeout'])
        return zone

    def _delete(self, zone):
        self.conn.dns.delete_zone(zone.id)
        for count in self.sdk.utils.iterate_timeout(timeout=self.params['timeout'], message='Timeout waiting for zone to be absent'):
            if self.conn.dns.find_zone(zone.id) is None:
                break

    def _update(self, zone, update):
        attributes = update.get('attributes')
        if attributes:
            zone = self.conn.dns.update_zone(zone.id, **attributes)
        if self.params['wait']:
            self.sdk.resource.wait_for_status(self.conn.dns, zone, status='active', failures=['error'], wait=self.params['timeout'])
        return zone

    def _will_change(self, state, zone):
        if state == 'present' and (not zone):
            return True
        elif state == 'present' and zone:
            return bool(self._build_update(zone))
        elif state == 'absent' and zone:
            return True
        else:
            return False