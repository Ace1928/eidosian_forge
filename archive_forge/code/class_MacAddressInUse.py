from oslo_utils import excutils
from neutron_lib._i18n import _
class MacAddressInUse(InUse):
    """An network operational error indicating a MAC address is already in use.

    A specialization of the InUse exception indicating an operation failed
    on a network because a specified MAC address is already in use on that
    network.

    :param net_id: The UUID of the network.
    :param mac: The requested MAC address that's already in use.
    """
    message = _('Unable to complete operation for network %(net_id)s. The mac address %(mac)s is in use.')