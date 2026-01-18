import unittest
from traits.api import Delegate, HasTraits, Instance, Str
def _s_changed(self, name, old, new):
    global baz_s_handler_self
    baz_s_handler_self = self