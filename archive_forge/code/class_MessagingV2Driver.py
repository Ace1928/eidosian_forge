import logging
import oslo_messaging
from oslo_messaging.notify import notifier
class MessagingV2Driver(MessagingDriver):
    """Send notifications using the 2.0 message format."""

    def __init__(self, conf, **kwargs):
        super(MessagingV2Driver, self).__init__(conf, version=2.0, **kwargs)