from neutron_lib._i18n import _
from neutron_lib import exceptions
class FirewallRuleNotFound(exceptions.NotFound):
    message = _('Firewall rule %(firewall_rule_id)s could not be found.')