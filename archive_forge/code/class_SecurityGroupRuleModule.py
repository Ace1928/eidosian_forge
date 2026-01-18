from ansible_collections.openstack.cloud.plugins.module_utils.openstack import (
class SecurityGroupRuleModule(OpenStackModule):
    argument_spec = dict(description=dict(), direction=dict(default='ingress', choices=['egress', 'ingress']), ether_type=dict(default='IPv4', choices=['IPv4', 'IPv6'], aliases=['ethertype']), port_range_max=dict(type='int'), port_range_min=dict(type='int'), project=dict(), protocol=dict(), remote_group=dict(), remote_ip_prefix=dict(), security_group=dict(required=True), state=dict(default='present', choices=['absent', 'present']))
    module_kwargs = dict(mutually_exclusive=[['remote_ip_prefix', 'remote_group']], supports_check_mode=True)

    def run(self):
        state = self.params['state']
        security_group_rule = self._find()
        if self.ansible.check_mode:
            self.exit_json(changed=self._will_change(state, security_group_rule))
        if state == 'present' and (not security_group_rule):
            security_group_rule = self._create()
            self.exit_json(changed=True, rule=security_group_rule.to_dict(computed=False))
        elif state == 'present' and security_group_rule:
            self.exit_json(changed=False, rule=security_group_rule.to_dict(computed=False))
        elif state == 'absent' and security_group_rule:
            self._delete(security_group_rule)
            self.exit_json(changed=True)
        elif state == 'absent' and (not security_group_rule):
            self.exit_json(changed=False)

    def _create(self):
        prototype = self._define_prototype()
        return self.conn.network.create_security_group_rule(**prototype)

    def _define_prototype(self):
        filters = {}
        prototype = dict(((k, self.params[k]) for k in ['description', 'direction', 'remote_ip_prefix'] if self.params[k] is not None))
        project_name_or_id = self.params['project']
        if project_name_or_id is not None:
            project = self.conn.identity.find_project(project_name_or_id, ignore_missing=False)
            filters = {'project_id': project.id}
            prototype['project_id'] = project.id
        security_group_name_or_id = self.params['security_group']
        security_group = self.conn.network.find_security_group(security_group_name_or_id, ignore_missing=False, **filters)
        prototype['security_group_id'] = security_group.id
        remote_group = None
        remote_group_name_or_id = self.params['remote_group']
        if remote_group_name_or_id is not None:
            remote_group = self.conn.network.find_security_group(remote_group_name_or_id, ignore_missing=False)
            prototype['remote_group_id'] = remote_group.id
        ether_type = self.params['ether_type']
        if ether_type is not None:
            prototype['ether_type'] = ether_type
        protocol = self.params['protocol']
        if protocol is not None and protocol not in ['any', '0']:
            prototype['protocol'] = protocol
        port_range_max = self.params['port_range_max']
        port_range_min = self.params['port_range_min']
        if protocol in ['icmp', 'ipv6-icmp']:
            if port_range_max is not None and int(port_range_max) != -1:
                prototype['port_range_max'] = int(port_range_max)
            if port_range_min is not None and int(port_range_min) != -1:
                prototype['port_range_min'] = int(port_range_min)
        elif protocol in ['tcp', 'udp']:
            if port_range_max is not None and int(port_range_max) != -1:
                prototype['port_range_max'] = int(port_range_max)
            if port_range_min is not None and int(port_range_min) != -1:
                prototype['port_range_min'] = int(port_range_min)
        elif protocol in ['any', '0']:
            pass
        else:
            if port_range_max is not None:
                prototype['port_range_max'] = int(port_range_max)
            if port_range_min is not None:
                prototype['port_range_min'] = int(port_range_min)
        return prototype

    def _delete(self, security_group_rule):
        self.conn.network.delete_security_group_rule(security_group_rule.id)

    def _find(self):
        matches = self._find_matches()
        if len(matches) > 1:
            self.fail_json(msg='Found more a single matching security group rule which match the given parameters.')
        elif len(matches) == 1:
            return self.conn.network.get_security_group_rule(matches[0]['id'])
        else:
            return None

    def _find_matches(self):
        prototype = self._define_prototype()
        security_group = self.conn.network.get_security_group(prototype['security_group_id'])
        if 'ether_type' in prototype:
            prototype['ethertype'] = prototype.pop('ether_type')
        if 'protocol' in prototype and prototype['protocol'] in ['tcp', 'udp']:
            if 'port_range_max' in prototype and prototype['port_range_max'] in [-1, 65535]:
                prototype.pop('port_range_max')
            if 'port_range_min' in prototype and prototype['port_range_min'] in [-1, 1]:
                prototype.pop('port_range_min')
        return [r for r in security_group.security_group_rules if all((r[k] == prototype[k] for k in prototype.keys()))]

    def _will_change(self, state, security_group_rule):
        if state == 'present' and (not security_group_rule):
            return True
        elif state == 'present' and security_group_rule:
            return False
        elif state == 'absent' and security_group_rule:
            return True
        else:
            return False