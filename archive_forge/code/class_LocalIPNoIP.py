from neutron_lib._i18n import _
from neutron_lib import exceptions
class LocalIPNoIP(exceptions.InvalidInput):
    message = _('Specified Port %(port_id)s has no fixed IPs configured.')