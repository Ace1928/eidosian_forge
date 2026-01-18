from neutron_lib._i18n import _
from neutron_lib import exceptions
class FirewallRuleInvalidAction(exceptions.InvalidInput):
    message = _('Action %(action)s is not supported. Only action values %(values)s are supported.')