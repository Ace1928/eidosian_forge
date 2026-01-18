import sys
from os_win._i18n import _
class HyperVvSwitchNotFound(NotFound, HyperVException):
    msg_fmt = _('vSwitch not found: %(vswitch_name)s.')