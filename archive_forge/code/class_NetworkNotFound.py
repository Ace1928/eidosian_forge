from oslo_utils import excutils
from neutron_lib._i18n import _
class NetworkNotFound(NotFound):
    """An exception indicating a network was not found.

    A specialization of the NotFound exception indicating a requested network
    could not be found.

    :param net_id: The UUID of the (not found) network requested.
    """
    message = _('Network %(net_id)s could not be found.')