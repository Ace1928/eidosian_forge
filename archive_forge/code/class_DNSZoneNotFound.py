import sys
from os_win._i18n import _
class DNSZoneNotFound(NotFound, DNSException):
    msg_fmt = _('DNS Zone not found: %(zone_name)s')