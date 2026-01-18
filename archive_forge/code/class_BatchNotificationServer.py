import itertools
import logging
from oslo_messaging.notify import dispatcher as notify_dispatcher
from oslo_messaging import server as msg_server
from oslo_messaging import transport as msg_transport
class BatchNotificationServer(NotificationServerBase):

    def _process_incoming(self, incoming):
        try:
            not_processed_messages = self.dispatcher.dispatch(incoming)
        except Exception:
            not_processed_messages = set(incoming)
            LOG.exception('Exception during messages handling.')
        for m in incoming:
            try:
                if m in not_processed_messages and self._allow_requeue:
                    m.requeue()
                else:
                    m.acknowledge()
            except Exception:
                LOG.exception('Fail to ack/requeue message.')