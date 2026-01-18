from neutron_lib._i18n import _
from neutron_lib import exceptions
class MissingEndpointForEndpointGroup(exceptions.BadRequest):
    message = _("No endpoints specified for endpoint group '%(group)s'")