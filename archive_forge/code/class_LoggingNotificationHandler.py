import logging
from oslo_config import cfg
from oslo_messaging.notify import notifier
class LoggingNotificationHandler(logging.Handler):
    """Handler for logging to the messaging notification system.

    Each time the application logs a message using the :py:mod:`logging`
    module, it will be sent as a notification. The severity used for the
    notification will be the same as the one used for the log record.

    This can be used into a Python logging configuration this way::

      [handler_notifier]
      class=oslo_messaging.LoggingNotificationHandler
      level=ERROR
      args=('rabbit:///')

    """
    CONF = cfg.CONF
    'Default configuration object used, subclass this class if you want to\n    use another one.\n\n    '

    def __init__(self, url, publisher_id=None, driver=None, topic=None, serializer=None):
        self.notifier = notifier.Notifier(notifier.get_notification_transport(self.CONF, url), publisher_id, driver, serializer() if serializer else None, topics=topic if isinstance(topic, list) or topic is None else [topic])
        logging.Handler.__init__(self)

    def emit(self, record):
        """Emit the log record to the messaging notification system.

        :param record: A log record to emit.

        """
        method = getattr(self.notifier, record.levelname.lower(), None)
        if not method:
            return
        method(None, 'logrecord', {'name': record.name, 'levelno': record.levelno, 'levelname': record.levelname, 'exc_info': record.exc_info, 'pathname': record.pathname, 'lineno': record.lineno, 'msg': record.getMessage(), 'funcName': record.funcName, 'thread': record.thread, 'processName': record.processName, 'process': record.process, 'extra': getattr(record, 'extra', None)})