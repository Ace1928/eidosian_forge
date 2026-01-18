from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
import enum
def EventsLabels(self, provider):
    return (e.label for e in self.Provider(provider).events)