import sys
from os_win._i18n import _
class VHDResizeException(HyperVException):
    msg_fmt = _('Exception encountered while resizing the VHD %(vhd_path)s.Reason: %(reason)s')