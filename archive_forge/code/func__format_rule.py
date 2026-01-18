from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
def _format_rule(self, r):
    rule = dict(r)
    rule['security_group_id'] = self.resource_id
    if 'remote_mode' in rule:
        remote_mode = rule.get(self.RULE_REMOTE_MODE)
        del rule[self.RULE_REMOTE_MODE]
        if remote_mode == self.RULE_REMOTE_GROUP_ID:
            rule[self.RULE_REMOTE_IP_PREFIX] = None
            if not rule.get(self.RULE_REMOTE_GROUP_ID):
                rule[self.RULE_REMOTE_GROUP_ID] = self.resource_id
        else:
            rule[self.RULE_REMOTE_GROUP_ID] = None
    for key in (self.RULE_PORT_RANGE_MIN, self.RULE_PORT_RANGE_MAX):
        if rule.get(key) is not None:
            rule[key] = str(rule[key])
    return rule