from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
import enum
def ProviderForEvent(self, event_label):
    for p in self.providers:
        if event_label in self.EventsLabels(p.label):
            return p
    return self._unadvertised_provider