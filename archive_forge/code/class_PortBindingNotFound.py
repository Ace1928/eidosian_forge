from oslo_utils import excutils
from neutron_lib._i18n import _
class PortBindingNotFound(NotFound):
    message = _('Binding for port %(port_id)s for host %(host)s could not be found.')