import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def _anytrait_changed(self, trait_name, old, new):
    if trait_name == 'value1':
        self.value1_count += 1
    elif trait_name == 'value2':
        self.value2_count += 1