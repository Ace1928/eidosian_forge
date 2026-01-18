import unittest
from traits.api import Delegate, HasTraits, Instance, Str
def _u_changed(self, name, old, new):
    global baz_u_handler_self
    baz_u_handler_self = self