import unittest
from traits.api import (
from traits.observation.api import (
def _dummy2_default(self):
    return Dummy2(dummy=self.dummy1)