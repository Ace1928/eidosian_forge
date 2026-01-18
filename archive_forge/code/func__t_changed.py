import unittest
from traits.api import Delegate, HasTraits, Instance, Str
def _t_changed(self, name, old, new):
    global baz_t_handler_self
    baz_t_handler_self = self