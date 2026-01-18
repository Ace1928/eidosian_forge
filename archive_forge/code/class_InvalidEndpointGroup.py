from neutron_lib._i18n import _
from neutron_lib import exceptions
class InvalidEndpointGroup(exceptions.BadRequest):
    message = _('Endpoint group%(suffix)s %(which)s cannot be specified, when VPN Service has subnet specified')