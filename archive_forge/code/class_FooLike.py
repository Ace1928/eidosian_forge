import abc
import unittest
import warnings
from traits.api import ABCHasTraits, ABCMetaHasTraits, HasTraits, Int, Float
class FooLike(HasTraits):
    x = Int(10)
    y = Float(20.0)

    def foo(self):
        return 'foo'

    @property
    def bar(self):
        return 'bar'