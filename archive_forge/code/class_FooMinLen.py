import unittest
from traits.api import CList, HasTraits, Instance, Int, List, Str, TraitError
class FooMinLen(HasTraits):
    l = List(Str, minlen=1)