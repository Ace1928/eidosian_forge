import unittest
from traits.api import HasTraits, Subclass, TraitError, Type
class ExampleTypeModel(HasTraits):
    _class = Type(klass=BaseClass)