from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
import enum
def AllEventLabels(self):
    all_events = (self.EventsLabels(p.label) for p in self.providers)
    return itertools.chain.from_iterable(all_events)