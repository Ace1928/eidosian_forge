from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.qos import rule as qos_rule
class CreateQoSDscpMarkingRule(qos_rule.QosRuleMixin, neutronv20.CreateCommand):
    """Create a QoS DSCP marking rule."""
    resource = DSCP_MARKING_RESOURCE

    def add_known_arguments(self, parser):
        super(CreateQoSDscpMarkingRule, self).add_known_arguments(parser)
        add_dscp_marking_arguments(parser)

    def args2body(self, parsed_args):
        body = {}
        update_dscp_args2body(parsed_args, body)
        return {self.resource: body}