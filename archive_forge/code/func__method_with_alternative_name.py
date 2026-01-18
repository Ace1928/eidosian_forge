import unittest
from traits.api import (
from traits.observation.api import (
@observe('foo')
def _method_with_alternative_name(self, foo_change_event):
    self.call_count += 1