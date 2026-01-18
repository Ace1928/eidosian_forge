from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class DnsRecordsetModule(OpenStackModule):
    argument_spec = dict(description=dict(), name=dict(required=True), records=dict(type='list', elements='str'), recordset_type=dict(choices=['a', 'aaaa', 'mx', 'cname', 'txt', 'ns', 'srv', 'ptr', 'caa']), state=dict(default='present', choices=['absent', 'present']), ttl=dict(type='int'), zone=dict(required=True))
    module_kwargs = dict(required_if=[('state', 'present', ['recordset_type', 'records'])], supports_check_mode=True)
    module_min_sdk_version = '0.28.0'

    def _needs_update(self, params, recordset):
        if params['records'] is not None:
            params['records'] = sorted(params['records'])
        if recordset['records'] is not None:
            recordset['records'] = sorted(recordset['records'])
        for k in ('description', 'records', 'ttl', 'type'):
            if k not in params:
                continue
            if k not in recordset:
                return True
            if params[k] is not None and params[k] != recordset[k]:
                return True
        return False

    def _system_state_change(self, state, recordset):
        if state == 'present':
            if recordset is None:
                return True
            kwargs = self._build_params()
            return self._needs_update(kwargs, recordset)
        if state == 'absent' and recordset:
            return True
        return False

    def _build_params(self):
        recordset_type = self.params['recordset_type']
        records = self.params['records']
        description = self.params['description']
        ttl = self.params['ttl']
        params = {'description': description, 'records': records, 'type': recordset_type.upper(), 'ttl': ttl}
        return {k: v for k, v in params.items() if v is not None}

    def run(self):
        zone = self.params.get('zone')
        name = self.params.get('name')
        state = self.params.get('state')
        ttl = self.params.get('ttl')
        zone = self.conn.dns.find_zone(name_or_id=zone, ignore_missing=False)
        recordset = self.conn.dns.find_recordset(zone, name)
        if self.ansible.check_mode:
            self.exit_json(changed=self._system_state_change(state, recordset))
        changed = False
        if state == 'present':
            kwargs = self._build_params()
            if recordset is None:
                kwargs['ttl'] = ttl or 300
                recordset = self.conn.dns.create_recordset(zone, name=name, **kwargs)
                changed = True
            elif self._needs_update(kwargs, recordset):
                recordset = self.conn.dns.update_recordset(recordset, **kwargs)
                changed = True
            self.exit_json(changed=changed, recordset=recordset)
        elif state == 'absent' and recordset is not None:
            self.conn.dns.delete_recordset(recordset)
            changed = True
        self.exit_json(changed=changed)