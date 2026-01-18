import urllib.parse
from glance_store.i18n import _
class GlanceStoreException(Exception):
    """
    Base Glance Store Exception

    To correctly use this class, inherit from it and define
    a 'message' property. That message will get printf'd
    with the keyword arguments provided to the constructor.
    """
    message = _('An unknown exception occurred')

    def __init__(self, message=None, **kwargs):
        if not message:
            message = self.message
        try:
            if kwargs:
                message = message % kwargs
        except Exception:
            pass
        self.msg = message
        super(GlanceStoreException, self).__init__(message)