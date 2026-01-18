from oslo_utils import excutils
from neutron_lib._i18n import _
class GatewayIpInUse(InUse):
    message = _('Current gateway ip %(ip_address)s already in use by port %(port_id)s. Unable to update.')