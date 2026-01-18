import sys
from os_win._i18n import _
class HyperVPortNotFoundException(NotFound, HyperVException):
    msg_fmt = _('Switch port not found: %(port_name)s')