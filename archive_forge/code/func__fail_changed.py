import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, List
def _fail_changed(self, name, old, new):
    raise Exception('error')