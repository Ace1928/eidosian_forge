from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.qos import policy as qos_policy
class ListQoSRuleTypes(neutronv20.ListCommand):
    """List available qos rule types."""
    resource = 'rule_type'
    shadow_resource = 'qos_rule_type'
    pagination_support = True
    sorting_support = True