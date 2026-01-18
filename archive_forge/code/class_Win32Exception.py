import sys
from os_win._i18n import _
class Win32Exception(OSWinException):
    msg_fmt = _('Executing Win32 API function %(func_name)s failed. Error code: %(error_code)s. Error message: %(error_message)s')