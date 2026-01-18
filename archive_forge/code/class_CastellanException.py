import urllib
from castellan.i18n import _
class CastellanException(Exception):
    """Base Castellan Exception

    To correctly use this class, inherit from it and define
    a 'message' property. That message will get printf'd
    with the keyword arguments provided to the constructor.
    """
    message = _('An unknown exception occurred')

    def __init__(self, message_arg=None, *args, **kwargs):
        if not message_arg:
            message_arg = self.message
        try:
            self.message = message_arg % kwargs
        except Exception:
            if _FATAL_EXCEPTION_FORMAT_ERRORS:
                raise
            else:
                pass
        super(CastellanException, self).__init__(self.message)