from oslo_utils import excutils
from neutron_lib._i18n import _
class InUse(NeutronException):
    """A generic exception indicating a resource is already in use."""
    message = _('The resource is in use.')