import unittest
from traits.api import HasTraits, Instance, Str, Any, Property
class BarInstance(HasTraits):
    other = Instance('BazInstance', copy='ref')
    unique = Instance(Foo)
    shared = Instance(Foo)
    ref = Instance(Foo, copy='ref')