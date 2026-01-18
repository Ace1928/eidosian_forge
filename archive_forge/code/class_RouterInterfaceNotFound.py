from neutron_lib._i18n import _
from neutron_lib import exceptions
class RouterInterfaceNotFound(exceptions.NotFound):
    message = _('Router %(router_id)s does not have an interface with id %(port_id)s')