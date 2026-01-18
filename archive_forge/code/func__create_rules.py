from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
def _create_rules(self, rules):
    egress_deleted = False
    for i in rules:
        if i[self.RULE_DIRECTION] == 'egress' and (not egress_deleted):
            egress_deleted = True

            def is_egress(rule):
                return rule[self.RULE_DIRECTION] == 'egress'
            self._delete_rules(is_egress)
        rule = self._format_rule(i)
        try:
            self.client().create_security_group_rule({'security_group_rule': rule})
        except Exception as ex:
            if not self.client_plugin().is_conflict(ex):
                raise