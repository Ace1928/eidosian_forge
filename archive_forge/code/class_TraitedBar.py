import abc
import unittest
import warnings
from traits.api import ABCHasTraits, ABCMetaHasTraits, HasTraits, Int, Float
class TraitedBar(HasTraits, AbstractBar, metaclass=ABCMetaHasTraits):
    x = Int(10)

    def bar(self):
        return 'bar'