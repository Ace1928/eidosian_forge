import unittest
from traits.api import (
from traits.observation.api import (
def _records_default(self):
    self.records_default_call_count += 1
    return [Record()]