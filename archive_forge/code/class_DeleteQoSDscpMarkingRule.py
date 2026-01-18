from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.qos import rule as qos_rule
class DeleteQoSDscpMarkingRule(qos_rule.QosRuleMixin, neutronv20.DeleteCommand):
    """Delete a given qos dscp marking rule."""
    allow_names = False
    resource = DSCP_MARKING_RESOURCE