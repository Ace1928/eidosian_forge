from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import threading
from googlecloudsdk.core.console import console_attr
import six
def UpdateMessage(self, message, new_message):
    """Updates the message of the given MultilineConsoleMessage."""
    if not message:
        raise ValueError('A message must be passed.')
    if message not in self._messages:
        raise ValueError('The given message does not belong to this output object.')
    with self._lock:
        message._UpdateMessage(new_message)
        self._may_have_update = True