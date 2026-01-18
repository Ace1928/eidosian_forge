import logging
import oslo_messaging
from oslo_messaging.notify import notifier
class MessagingDriver(notifier.Driver):
    """Send notifications using the 1.0 message format.

    This driver sends notifications over the configured messaging transport,
    but without any message envelope (also known as message format 1.0).

    This driver should only be used in cases where there are existing consumers
    deployed which do not support the 2.0 message format.
    """

    def __init__(self, conf, topics, transport, version=1.0):
        super(MessagingDriver, self).__init__(conf, topics, transport)
        self.version = version

    def notify(self, ctxt, message, priority, retry):
        priority = priority.lower()
        for topic in self.topics:
            target = oslo_messaging.Target(topic='%s.%s' % (topic, priority))
            try:
                self.transport._send_notification(target, ctxt, message, version=self.version, retry=retry)
            except Exception:
                LOG.exception('Could not send notification to %(topic)s. Payload=%(message)s', {'topic': topic, 'message': message})