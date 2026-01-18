from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def _build_update_security_group_rules(self, security_group):
    if self.params['security_group_rules'] is None:
        return {}

    def find_security_group_rule_match(prototype, security_group_rules):
        matches = [r for r in security_group_rules if is_security_group_rule_match(prototype, r)]
        if len(matches) > 1:
            self.fail_json(msg='Found more a single matching security group rule which match the given parameters.')
        elif len(matches) == 1:
            return matches[0]
        else:
            return None

    def is_security_group_rule_match(prototype, security_group_rule):
        skip_keys = ['ether_type']
        if 'ether_type' in prototype and security_group_rule['ethertype'] != prototype['ether_type']:
            return False
        if 'protocol' in prototype and prototype['protocol'] in ['tcp', 'udp']:
            if 'port_range_max' in prototype and prototype['port_range_max'] in [-1, 65535]:
                if security_group_rule['port_range_max'] is not None:
                    return False
                skip_keys.append('port_range_max')
            if 'port_range_min' in prototype and prototype['port_range_min'] in [-1, 1]:
                if security_group_rule['port_range_min'] is not None:
                    return False
                skip_keys.append('port_range_min')
        if all((security_group_rule[k] == prototype[k] for k in set(prototype.keys()) - set(skip_keys))):
            return security_group_rule
        else:
            return None
    update = {}
    keep_security_group_rules = {}
    create_security_group_rules = []
    delete_security_group_rules = []
    for prototype in self._generate_security_group_rules(security_group):
        match = find_security_group_rule_match(prototype, security_group.security_group_rules)
        if match:
            keep_security_group_rules[match['id']] = match
        else:
            create_security_group_rules.append(prototype)
    for security_group_rule in security_group.security_group_rules:
        if security_group_rule['id'] not in keep_security_group_rules.keys():
            delete_security_group_rules.append(security_group_rule)
    if create_security_group_rules:
        update['create_security_group_rules'] = create_security_group_rules
    if delete_security_group_rules:
        update['delete_security_group_rules'] = delete_security_group_rules
    return update