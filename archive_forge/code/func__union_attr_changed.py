import unittest
from traits.api import (
def _union_attr_changed(self, new):
    self.shadow_union_trait = new