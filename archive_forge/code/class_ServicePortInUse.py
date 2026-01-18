from oslo_utils import excutils
from neutron_lib._i18n import _
class ServicePortInUse(InUse):
    """An error indicating a service port can't be deleted.

    A specialization of the InUse exception indicating a requested service
    port can't be deleted via the APIs.

    :param port_id: The UUID of the port requested.
    :param reason: Details on why the operation failed.
    """
    message = _('Port %(port_id)s cannot be deleted directly via the port API: %(reason)s.')