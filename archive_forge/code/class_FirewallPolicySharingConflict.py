from neutron_lib._i18n import _
from neutron_lib import exceptions
class FirewallPolicySharingConflict(exceptions.Conflict):
    """FWaaS exception raised for sharing policies

    Raised if policy is 'shared' but its associated rules are not.
    """
    message = _('Operation cannot be performed. Before sharing firewall policy %(firewall_policy_id)s, share associated firewall rule %(firewall_rule_id)s.')