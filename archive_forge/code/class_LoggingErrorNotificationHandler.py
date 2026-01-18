import logging
from oslo_config import cfg
class LoggingErrorNotificationHandler(logging.Handler):

    def __init__(self, *args, **kwargs):
        import oslo_messaging
        logging.Handler.__init__(self, *args, **kwargs)
        self._transport = oslo_messaging.get_notification_transport(cfg.CONF)
        self._notifier = oslo_messaging.Notifier(self._transport, publisher_id='error.publisher')

    def emit(self, record):
        conf = self._transport.conf
        if 'log' in conf.oslo_messaging_notifications.driver:
            return
        self._notifier.error({}, 'error_notification', dict(error=record.msg))