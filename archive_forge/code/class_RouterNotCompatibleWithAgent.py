from neutron_lib._i18n import _
from neutron_lib import exceptions
class RouterNotCompatibleWithAgent(exceptions.NeutronException):
    message = _("Router '%(router_id)s' is not compatible with this agent.")