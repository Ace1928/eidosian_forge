import itertools
import logging
import operator
from oslo_messaging import dispatcher
from oslo_messaging import serializer as msg_serializer
class BatchNotificationDispatcher(NotificationDispatcher):
    """A message dispatcher which understands Notification messages.

    A MessageHandlingServer is constructed by passing a callable dispatcher
    which is invoked with a list of message dictionaries each time 'batch_size'
    messages are received or 'batch_timeout' seconds is reached.
    """

    def dispatch(self, incoming):
        """Dispatch notification messages to the appropriate endpoint method.
        """
        messages_grouped = itertools.groupby(sorted((self._extract_user_message(m) for m in incoming), key=operator.itemgetter(0)), operator.itemgetter(0))
        requeues = set()
        for priority, messages in messages_grouped:
            __, raw_messages, messages = zip(*messages)
            if priority not in PRIORITIES:
                LOG.warning('Unknown priority "%s"', priority)
                continue
            for screen, callback in self._callbacks_by_priority.get(priority, []):
                if screen:
                    filtered_messages = [message for message in messages if screen.match(message['ctxt'], message['publisher_id'], message['event_type'], message['metadata'], message['payload'])]
                else:
                    filtered_messages = list(messages)
                if not filtered_messages:
                    continue
                ret = self._exec_callback(callback, filtered_messages)
                if ret == NotificationResult.REQUEUE:
                    requeues.update(raw_messages)
                    break
        return requeues

    def _exec_callback(self, callback, messages):
        try:
            return callback(messages)
        except Exception:
            LOG.exception('Callback raised an exception.')
            return NotificationResult.REQUEUE