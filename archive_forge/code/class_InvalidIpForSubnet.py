from oslo_utils import excutils
from neutron_lib._i18n import _
class InvalidIpForSubnet(BadRequest):
    """An exception indicating an invalid IP was specified for a subnet.

    A specialization of the BadRequest exception indicating a specified IP
    address is invalid for a subnet.

    :param ip_address: The IP address that's invalid on the subnet.
    """
    message = _('IP address %(ip_address)s is not a valid IP for the specified subnet.')