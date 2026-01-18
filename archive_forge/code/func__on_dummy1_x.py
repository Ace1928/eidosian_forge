import unittest
from traits.api import (
from traits.observation.api import (
@observe('dummy1.x')
def _on_dummy1_x(self, event):
    self.handler_called = True