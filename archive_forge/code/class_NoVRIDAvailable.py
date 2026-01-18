from neutron_lib._i18n import _
from neutron_lib import exceptions
class NoVRIDAvailable(exceptions.Conflict):
    message = _('No more Virtual Router Identifier (VRID) available when creating router %(router_id)s. The limit of number of HA Routers per tenant is 254.')