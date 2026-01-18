from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.qos import rule as qos_rule
class CreateQoSBandwidthLimitRule(qos_rule.QosRuleMixin, neutronv20.CreateCommand):
    """Create a qos bandwidth limit rule."""
    resource = BANDWIDTH_LIMIT_RULE_RESOURCE

    def add_known_arguments(self, parser):
        super(CreateQoSBandwidthLimitRule, self).add_known_arguments(parser)
        add_bandwidth_limit_arguments(parser)

    def args2body(self, parsed_args):
        body = {}
        update_bandwidth_limit_args2body(parsed_args, body)
        return {self.resource: body}