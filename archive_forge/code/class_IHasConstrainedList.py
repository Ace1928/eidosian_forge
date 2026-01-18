import unittest
from traits.api import CList, HasTraits, Instance, Int, List, Str, TraitError
class IHasConstrainedList(HasTraits):
    foo = List(Str, minlen=3)