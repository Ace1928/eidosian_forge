import unittest
from traits.api import (
from traits.observation.api import (
class ClassWithListOfListOfInstance(HasTraits):
    list_of_list_of_instances = List(List(Instance(SingleValue)))