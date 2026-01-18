from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.qos import rule as qos_rule
class ListQoSDscpMarkingRules(qos_rule.QosRuleMixin, neutronv20.ListCommand):
    """List all QoS DSCP marking rules belonging to the specified policy."""
    _formatters = {}
    pagination_support = True
    sorting_support = True
    resource = DSCP_MARKING_RESOURCE