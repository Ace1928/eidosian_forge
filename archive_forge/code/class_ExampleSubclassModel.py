import unittest
from traits.api import HasTraits, Subclass, TraitError, Type
class ExampleSubclassModel(HasTraits):
    _class = Subclass(BaseClass)