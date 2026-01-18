from neutron_lib._i18n import _
from neutron_lib import exceptions
class FirewallPolicyInUse(exceptions.InUse):
    message = _('Firewall policy %(firewall_policy_id)s is being used.')