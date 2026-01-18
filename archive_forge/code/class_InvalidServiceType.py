from oslo_utils import excutils
from neutron_lib._i18n import _
class InvalidServiceType(InvalidInput):
    """An error due to an invalid service type.

    :param service_type: The service type that's invalid.
    """
    message = _('Invalid service type: %(service_type)s.')