from oslo_utils import excutils
from neutron_lib._i18n import _
class PortNotFoundOnNetwork(NotFound):
    """An exception for a requested port on a network that's not found.

    A specialization of the NotFound exception that indicates a specified
    port on a specified network doesn't exist.

    :param port_id: The UUID of the (not found) port that was requested.
    :param net_id: The UUID of the network that was requested for the port.
    """
    message = _('Port %(port_id)s could not be found on network %(net_id)s.')