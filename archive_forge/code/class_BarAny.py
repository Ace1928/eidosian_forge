import unittest
from traits.api import HasTraits, Instance, Str, Any, Property
class BarAny(HasTraits):
    other = Any