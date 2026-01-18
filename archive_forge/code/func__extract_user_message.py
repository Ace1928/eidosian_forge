import itertools
import logging
import operator
from oslo_messaging import dispatcher
from oslo_messaging import serializer as msg_serializer
def _extract_user_message(self, incoming):
    ctxt = self.serializer.deserialize_context(incoming.ctxt)
    message = incoming.message
    publisher_id = message.get('publisher_id')
    event_type = message.get('event_type')
    metadata = {'message_id': message.get('message_id'), 'timestamp': message.get('timestamp')}
    priority = message.get('priority', '').lower()
    payload = self.serializer.deserialize_entity(ctxt, message.get('payload'))
    return (priority, incoming, dict(ctxt=ctxt, publisher_id=publisher_id, event_type=event_type, payload=payload, metadata=metadata))