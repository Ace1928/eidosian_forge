import unittest
from traits.api import (
from traits.observation.api import (
class ClassWithDictOfInstance(HasTraits):
    name_to_instance = Dict(Str, Instance(SingleValue))