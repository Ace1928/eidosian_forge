from neutron_lib._i18n import _
from neutron_lib import exceptions
class FirewallPolicyConflict(exceptions.NotFound):
    """FWaaS exception raised for firewall policy conflict

    Raised when user tries to use another project's unshared policy.
    """
    message = _('Operation cannot be performed since firewall policy %(firewall_policy_id)s for your project could not be found. Please confirm if the firewall policy exists and is shared.')