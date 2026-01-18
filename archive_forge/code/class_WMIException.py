import sys
from os_win._i18n import _
class WMIException(OSWinException):

    def __init__(self, message=None, wmi_exc=None):
        if wmi_exc:
            try:
                wmi_exc_message = wmi_exc.com_error.excepinfo[2].strip()
                message = '%s WMI exception message: %s' % (message, wmi_exc_message)
            except AttributeError:
                pass
            except IndexError:
                pass
        super(WMIException, self).__init__(message)