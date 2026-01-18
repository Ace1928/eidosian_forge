import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, on_trait_change
def _on_foo_fuz_changed(obj, name, old, new):
    raise FuzException('function')