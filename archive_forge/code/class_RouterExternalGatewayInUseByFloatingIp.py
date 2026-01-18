from neutron_lib._i18n import _
from neutron_lib import exceptions
class RouterExternalGatewayInUseByFloatingIp(exceptions.InUse):
    message = _('Gateway cannot be updated for router %(router_id)s, since a gateway to external network %(net_id)s is required by one or more floating IPs.')