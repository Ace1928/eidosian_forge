from neutron_lib._i18n import _
from neutron_lib import exceptions
class RouterInterfaceNotFoundForSubnet(exceptions.NotFound):
    message = _('Router %(router_id)s has no interface on subnet %(subnet_id)s')