import logging
import warnings
from oslo_serialization import jsonutils
from oslo_utils import strutils
from oslo_messaging.notify import notifier
class LogDriver(notifier.Driver):
    """Publish notifications via Python logging infrastructure."""
    LOGGER_BASE = 'oslo.messaging.notification'

    def notify(self, ctxt, message, priority, retry):
        logger = logging.getLogger('%s.%s' % (self.LOGGER_BASE, message['event_type']))
        method = getattr(logger, priority.lower(), None)
        if method:
            method(jsonutils.dumps(strutils.mask_dict_password(message)))
        else:
            warnings.warn('Unable to log message as notify cannot find a logger with the priority specified %s' % priority.lower())