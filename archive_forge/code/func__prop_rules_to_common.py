from heat.common import exception
from heat.common.i18n import _
from heat.engine import properties
from heat.engine import resource
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