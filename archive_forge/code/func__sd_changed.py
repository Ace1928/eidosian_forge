import unittest
from traits.api import Delegate, HasTraits, Instance, Str
def _sd_changed(self, name, old, new):
    global baz_sd_handler_self
    baz_sd_handler_self = self