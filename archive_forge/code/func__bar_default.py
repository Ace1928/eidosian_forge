import unittest
from traits.trait_types import Int
from traits.has_traits import HasTraits
def _bar_default(self):
    raise KeyError()