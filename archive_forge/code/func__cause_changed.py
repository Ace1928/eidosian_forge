import unittest
from traits.api import HasTraits, Str, Instance, Any
def _cause_changed(self, obj, name, old, new):
    self.test.events_delivered.append('Baz._caused_changed')