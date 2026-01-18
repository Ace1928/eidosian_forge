from oslo_utils import excutils
from neutron_lib._i18n import _
class IpAddressGenerationFailure(Conflict):
    """A conflict error due to no more IP addresses on a said network.

    :param net_id: The UUID of the network that has no more IP addresses.
    """
    message = _('No more IP addresses available on network %(net_id)s.')