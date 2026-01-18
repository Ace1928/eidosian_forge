import unittest
from traits.api import (
from traits.observation.api import (
@observe('foo')
def _the_usual_signature(self, event):
    self.call_count += 1