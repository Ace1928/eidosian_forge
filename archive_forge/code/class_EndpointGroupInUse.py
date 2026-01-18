from neutron_lib._i18n import _
from neutron_lib import exceptions
class EndpointGroupInUse(exceptions.BadRequest):
    message = _('Endpoint group %(group_id)s is in use and cannot be deleted')