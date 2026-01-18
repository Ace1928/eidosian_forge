from oslo_utils import excutils
from neutron_lib._i18n import _
class InvalidCIDR(BadRequest):
    message = _('Invalid CIDR %(input)s given as IP prefix.')