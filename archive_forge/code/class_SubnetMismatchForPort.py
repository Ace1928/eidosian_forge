from oslo_utils import excutils
from neutron_lib._i18n import _
class SubnetMismatchForPort(BadRequest):
    """A bad request error indicating a specified subnet isn't on a port.

    A specialization of the BadRequest exception indicating a subnet on a port
    doesn't match a specified subnet.

    :param port_id: The UUID of the port.
    :param subnet_id: The UUID of the requested subnet.
    """
    message = _('Subnet on port %(port_id)s does not match the requested subnet %(subnet_id)s.')