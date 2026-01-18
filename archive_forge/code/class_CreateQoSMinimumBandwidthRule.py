from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.qos import rule as qos_rule
class CreateQoSMinimumBandwidthRule(qos_rule.QosRuleMixin, neutronv20.CreateCommand):
    """Create a qos minimum bandwidth rule."""
    resource = MINIMUM_BANDWIDTH_RULE_RESOURCE

    def add_known_arguments(self, parser):
        super(CreateQoSMinimumBandwidthRule, self).add_known_arguments(parser)
        add_minimum_bandwidth_arguments(parser)

    def args2body(self, parsed_args):
        body = {}
        update_minimum_bandwidth_args2body(parsed_args, body)
        return {self.resource: body}