from neutron_lib._i18n import _
from neutron_lib import exceptions
class InvalidEndpointInEndpointGroup(exceptions.InvalidInput):
    message = _("Endpoint '%(endpoint)s' is invalid for group type '%(group_type)s': %(why)s")