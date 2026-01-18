from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.qos import rule as qos_rule
class ShowQoSMinimumBandwidthRule(qos_rule.QosRuleMixin, neutronv20.ShowCommand):
    """Show information about the given qos minimum bandwidth rule."""
    resource = MINIMUM_BANDWIDTH_RULE_RESOURCE
    allow_names = False