from neutron_lib._i18n import _
from neutron_lib import exceptions
class FirewallRuleInvalidICMPParameter(exceptions.InvalidInput):
    message = _('%(param)s are not allowed when protocol is set to ICMP.')