from oslo_utils import excutils
from neutron_lib._i18n import _
class InvalidSubnetServiceType(InvalidInput):
    message = _('Subnet service type %(service_type)s does not correspond to a valid device owner.')