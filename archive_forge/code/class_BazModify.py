import unittest
from traits.api import Delegate, HasTraits, Instance, Str
class BazModify(HasTraits):
    foo = Instance(Foo, ())
    sd = Delegate('foo', prefix='s', modify=True)
    t = Delegate('foo', modify=True)
    u = Delegate('foo', listenable=False, modify=True)

    def _s_changed(self, name, old, new):
        global baz_s_handler_self
        baz_s_handler_self = self

    def _sd_changed(self, name, old, new):
        global baz_sd_handler_self
        baz_sd_handler_self = self

    def _t_changed(self, name, old, new):
        global baz_t_handler_self
        baz_t_handler_self = self

    def _u_changed(self, name, old, new):
        global baz_u_handler_self
        baz_u_handler_self = self