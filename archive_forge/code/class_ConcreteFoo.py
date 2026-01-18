import abc
import unittest
import warnings
from traits.api import ABCHasTraits, ABCMetaHasTraits, HasTraits, Int, Float
class ConcreteFoo(AbstractFoo):

    def foo(self):
        return 'foo'

    @property
    def bar(self):
        return 'bar'