from neutron_lib._i18n import _
from neutron_lib import exceptions
class NetworkAddressScopeAffinityError(exceptions.BadRequest):
    message = _('Subnets of the same address family hosted on the same network must all participate in the same address scope.')