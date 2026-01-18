from heat.common import exception
from heat.common.i18n import _
from heat.engine import properties
from heat.engine import resource
class NeutronSecurityGroup(object):

    def __init__(self, sg):
        self.sg = sg
        self.client = sg.client('neutron')
        self.plugin = sg.client_plugin('neutron')

    def _convert_to_neutron_rule(self, sg_rule):
        return {'direction': sg_rule['direction'], 'ethertype': 'IPv4', 'remote_ip_prefix': sg_rule.get(self.sg.RULE_CIDR_IP), 'port_range_min': sg_rule.get(self.sg.RULE_FROM_PORT), 'port_range_max': sg_rule.get(self.sg.RULE_TO_PORT), 'protocol': sg_rule.get(self.sg.RULE_IP_PROTOCOL), 'remote_group_id': sg_rule.get(self.sg.RULE_SOURCE_SECURITY_GROUP_ID), 'security_group_id': self.sg.resource_id}

    def _res_rules_to_common(self, api_rules):
        rules = {}
        for nr in api_rules:
            rule = {}
            rule[self.sg.RULE_FROM_PORT] = nr['port_range_min']
            rule[self.sg.RULE_TO_PORT] = nr['port_range_max']
            rule[self.sg.RULE_IP_PROTOCOL] = nr['protocol']
            rule['direction'] = nr['direction']
            rule[self.sg.RULE_CIDR_IP] = nr['remote_ip_prefix']
            rule[self.sg.RULE_SOURCE_SECURITY_GROUP_ID] = nr['remote_group_id']
            rules[nr['id']] = rule
        return rules

    def _prop_rules_to_common(self, props, direction):
        rules = []
        prs = props.get(direction) or []
        for pr in prs:
            rule = dict(pr)
            rule.pop(self.sg.RULE_SOURCE_SECURITY_GROUP_OWNER_ID)
            from_port = pr.get(self.sg.RULE_FROM_PORT)
            if from_port is not None:
                from_port = int(from_port)
                if from_port < 0:
                    from_port = None
            rule[self.sg.RULE_FROM_PORT] = from_port
            to_port = pr.get(self.sg.RULE_TO_PORT)
            if to_port is not None:
                to_port = int(to_port)
                if to_port < 0:
                    to_port = None
            rule[self.sg.RULE_TO_PORT] = to_port
            if pr.get(self.sg.RULE_FROM_PORT) is None and pr.get(self.sg.RULE_TO_PORT) is None:
                rule[self.sg.RULE_CIDR_IP] = None
            else:
                rule[self.sg.RULE_CIDR_IP] = pr.get(self.sg.RULE_CIDR_IP)
            rule[self.sg.RULE_SOURCE_SECURITY_GROUP_ID] = pr.get(self.sg.RULE_SOURCE_SECURITY_GROUP_ID) or pr.get(self.sg.RULE_SOURCE_SECURITY_GROUP_NAME)
            rule.pop(self.sg.RULE_SOURCE_SECURITY_GROUP_NAME)
            rules.append(rule)
        return rules

    def create(self):
        sec = self.client.create_security_group({'security_group': {'name': self.sg.physical_resource_name(), 'description': self.sg.properties[self.sg.GROUP_DESCRIPTION]}})['security_group']
        self.sg.resource_id_set(sec['id'])
        self.delete_default_egress_rules(sec)
        if self.sg.properties[self.sg.SECURITY_GROUP_INGRESS]:
            rules_in = self._prop_rules_to_common(self.sg.properties, self.sg.SECURITY_GROUP_INGRESS)
            for rule in rules_in:
                rule['direction'] = 'ingress'
                self.create_rule(rule)
        if self.sg.properties[self.sg.SECURITY_GROUP_EGRESS]:
            rules_e = self._prop_rules_to_common(self.sg.properties, self.sg.SECURITY_GROUP_EGRESS)
            for rule in rules_e:
                rule['direction'] = 'egress'
                self.create_rule(rule)

    def create_rule(self, rule):
        try:
            self.client.create_security_group_rule({'security_group_rule': self._convert_to_neutron_rule(rule)})
        except Exception as ex:
            if not self.plugin.is_conflict(ex):
                raise

    def delete(self):
        if self.sg.resource_id is not None:
            try:
                sec = self.client.show_security_group(self.sg.resource_id)['security_group']
            except Exception as ex:
                self.plugin.ignore_not_found(ex)
            else:
                for rule in sec['security_group_rules']:
                    self.delete_rule(rule['id'])
                with self.plugin.ignore_not_found:
                    self.client.delete_security_group(self.sg.resource_id)

    def delete_rule(self, rule_id):
        with self.plugin.ignore_not_found:
            self.client.delete_security_group_rule(rule_id)

    def delete_default_egress_rules(self, sec):
        """Delete the default rules which allow all egress traffic."""
        if self.sg.properties[self.sg.SECURITY_GROUP_EGRESS]:
            for rule in sec['security_group_rules']:
                if rule['direction'] == 'egress':
                    self.client.delete_security_group_rule(rule['id'])

    def update(self, props):
        sec = self.client.show_security_group(self.sg.resource_id)['security_group']
        existing = self._res_rules_to_common(sec['security_group_rules'])
        updated = {}
        updated[self.sg.SECURITY_GROUP_EGRESS] = self._prop_rules_to_common(props, self.sg.SECURITY_GROUP_EGRESS)
        updated[self.sg.SECURITY_GROUP_INGRESS] = self._prop_rules_to_common(props, self.sg.SECURITY_GROUP_INGRESS)
        ids, new = self.diff_rules(existing, updated)
        for id in ids:
            self.delete_rule(id)
        for rule in new:
            self.create_rule(rule)

    def diff_rules(self, existing, updated):
        for rule in updated[self.sg.SECURITY_GROUP_EGRESS]:
            rule['direction'] = 'egress'
        for rule in updated[self.sg.SECURITY_GROUP_INGRESS]:
            rule['direction'] = 'ingress'
        updated_rules = list(updated.values())
        updated_all = updated_rules[0] + updated_rules[1]
        ids_to_delete = [id for id, rule in existing.items() if rule not in updated_all]
        rules_to_create = [rule for rule in updated_all if rule not in existing.values()]
        return (ids_to_delete, rules_to_create)