from neutron_lib._i18n import _
from neutron_lib import exceptions
class LocalIPNoRequestedIP(exceptions.InvalidInput):
    message = _('Specified Port %(port_id)s has several IPs, should specify exact IP address to use for Local IP.')