from neutron_lib._i18n import _
from neutron_lib import exceptions
class UnableToRemoveGateways(exceptions.NeutronException):
    message = _('Unable to remove extra gateways from a router %(router_id)s')