import urllib.parse as urlparse
from glance.i18n import _
class LimitExceeded(GlanceException):
    message = _('The request returned a 413 Request Entity Too Large. This generally means that rate limiting or a quota threshold was breached.\n\nThe response body:\n%(body)s')

    def __init__(self, *args, **kwargs):
        self.retry_after = int(kwargs['retry']) if kwargs.get('retry') else None
        super(LimitExceeded, self).__init__(*args, **kwargs)