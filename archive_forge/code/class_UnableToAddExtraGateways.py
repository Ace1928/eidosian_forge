from neutron_lib._i18n import _
from neutron_lib import exceptions
class UnableToAddExtraGateways(exceptions.NeutronException):
    message = _('Unable to add extra gateways to a router %(router_id)s')