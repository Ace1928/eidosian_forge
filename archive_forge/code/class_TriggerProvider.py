from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
import enum
class TriggerProvider(object):
    """Represents --trigger-provider flag value options."""

    def __init__(self, label, events):
        self.label = label
        self.events = events
        for event in self.events:
            event.provider = self

    @property
    def default_event(self):
        return self.events[0]