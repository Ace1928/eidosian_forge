from oslo_utils import excutils
from neutron_lib._i18n import _
class IpAddressInUse(InUse):
    """An network operational error indicating an IP address is already in use.

    A specialization of the InUse exception indicating an operation can't
    complete because an IP address is in use.

    :param net_id: The UUID of the network.
    :param ip_address: The IP address that's already in use on the network.
    """
    message = _('Unable to complete operation for network %(net_id)s. The IP address %(ip_address)s is in use.')