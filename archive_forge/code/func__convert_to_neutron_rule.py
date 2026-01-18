from heat.common import exception
from heat.common.i18n import _
from heat.engine import properties
from heat.engine import resource
def _convert_to_neutron_rule(self, sg_rule):
    return {'direction': sg_rule['direction'], 'ethertype': 'IPv4', 'remote_ip_prefix': sg_rule.get(self.sg.RULE_CIDR_IP), 'port_range_min': sg_rule.get(self.sg.RULE_FROM_PORT), 'port_range_max': sg_rule.get(self.sg.RULE_TO_PORT), 'protocol': sg_rule.get(self.sg.RULE_IP_PROTOCOL), 'remote_group_id': sg_rule.get(self.sg.RULE_SOURCE_SECURITY_GROUP_ID), 'security_group_id': self.sg.resource_id}