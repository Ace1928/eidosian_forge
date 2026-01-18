import itertools
import logging
import operator
from oslo_messaging import dispatcher
from oslo_messaging import serializer as msg_serializer
class NotificationDispatcher(dispatcher.DispatcherBase):

    def __init__(self, endpoints, serializer):
        self.endpoints = endpoints
        self.serializer = serializer or msg_serializer.NoOpSerializer()
        self._callbacks_by_priority = {}
        for endpoint, prio in itertools.product(endpoints, PRIORITIES):
            if hasattr(endpoint, prio):
                method = getattr(endpoint, prio)
                screen = getattr(endpoint, 'filter_rule', None)
                self._callbacks_by_priority.setdefault(prio, []).append((screen, method))

    @property
    def supported_priorities(self):
        return self._callbacks_by_priority.keys()

    def dispatch(self, incoming):
        """Dispatch notification messages to the appropriate endpoint method.
        """
        priority, raw_message, message = self._extract_user_message(incoming)
        if priority not in PRIORITIES:
            LOG.warning('Unknown priority "%s"', priority)
            return
        for screen, callback in self._callbacks_by_priority.get(priority, []):
            if screen and (not screen.match(message['ctxt'], message['publisher_id'], message['event_type'], message['metadata'], message['payload'])):
                continue
            ret = self._exec_callback(callback, message)
            if ret == NotificationResult.REQUEUE:
                return ret
        return NotificationResult.HANDLED

    def _exec_callback(self, callback, message):
        try:
            return callback(message['ctxt'], message['publisher_id'], message['event_type'], message['payload'], message['metadata'])
        except Exception:
            LOG.exception('Callback raised an exception.')
            return NotificationResult.REQUEUE

    def _extract_user_message(self, incoming):
        ctxt = self.serializer.deserialize_context(incoming.ctxt)
        message = incoming.message
        publisher_id = message.get('publisher_id')
        event_type = message.get('event_type')
        metadata = {'message_id': message.get('message_id'), 'timestamp': message.get('timestamp')}
        priority = message.get('priority', '').lower()
        payload = self.serializer.deserialize_entity(ctxt, message.get('payload'))
        return (priority, incoming, dict(ctxt=ctxt, publisher_id=publisher_id, event_type=event_type, payload=payload, metadata=metadata))