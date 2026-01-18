from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from urllib import parse
def _reformat_properties(self, props):
    rule = {}
    for name in self.PROPERTIES:
        if name in props:
            rule[name] = props.pop(name)
    if rule:
        props['%s_rule' % self.alarm_type] = rule
    return props