from neutron_lib._i18n import _
from neutron_lib import exceptions
class FirewallRuleAlreadyAssociated(exceptions.Conflict):
    """FWaaS exception for an already associated rule

    Occurs when there is an attempt to assign a rule to a policy that
    the rule is already associated with.
    """
    message = _('Operation cannot be performed since firewall rule %(firewall_rule_id)s is already associated with firewall policy %(firewall_policy_id)s.')