import sys
from os_win._i18n import _
class OSWinException(Exception):
    msg_fmt = 'An exception has been encountered.'

    def __init__(self, message=None, **kwargs):
        self.kwargs = kwargs
        self.error_code = kwargs.get('error_code')
        if not message:
            message = self.msg_fmt % kwargs
        self.message = message
        super(OSWinException, self).__init__(message)