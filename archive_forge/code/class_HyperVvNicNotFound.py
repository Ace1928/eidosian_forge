import sys
from os_win._i18n import _
class HyperVvNicNotFound(NotFound, HyperVException):
    msg_fmt = _('vNic not found: %(vnic_name)s')