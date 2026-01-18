from oslo_utils import excutils
from neutron_lib._i18n import _
class PortInUse(InUse):
    """An operational error indicating a requested port is already attached.

    A specialization of the InUse exception indicating an operation failed on
    a port because it already has an attached device.

    :param port_id: The UUID of the port requested.
    :param net_id: The UUID of the requested port's network.
    :param device_id: The UUID of the device already attached to the port.
    """
    message = _('Unable to complete operation on port %(port_id)s for network %(net_id)s. Port already has an attached device %(device_id)s.')