from neutron_lib._i18n import _
from neutron_lib import exceptions
class LocalIPRequestedIPNotFound(exceptions.InvalidInput):
    message = _('Specified Port %(port_id)s does not have requested IP address: %(ip)s.')