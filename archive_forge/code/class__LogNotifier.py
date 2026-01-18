import os
import sys
class _LogNotifier(object):

    def __init__(self, log):
        self._log = log

    def notify(self, context, event_type, payload):
        self._log.info('Event type: %(event_type)s, Context: %(context)s, Payload: %(payload)s', {'context': context, 'event_type': event_type, 'payload': payload})