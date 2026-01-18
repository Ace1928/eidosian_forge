import os
import sys
class _MessagingNotifier(object):

    def __init__(self, notifier):
        self._notifier = notifier

    def notify(self, context, event_type, payload):
        self._notifier.info(context, event_type, payload)