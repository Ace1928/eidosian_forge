import sys
from os_win._i18n import _
class DNSZoneAlreadyExists(DNSException):
    msg_fmt = _('DNS Zone already exists: %(zone_name)s')