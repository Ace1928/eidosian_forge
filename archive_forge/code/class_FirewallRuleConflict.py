from neutron_lib._i18n import _
from neutron_lib import exceptions
class FirewallRuleConflict(exceptions.Conflict):
    """FWaaS rule conflict exception

    Occurs when admin policy tries to use another project's rule that is
    not shared.
    """
    message = _('Operation cannot be performed since firewall rule %(firewall_rule_id)s is not shared and belongs to another project %(project_id)s.')