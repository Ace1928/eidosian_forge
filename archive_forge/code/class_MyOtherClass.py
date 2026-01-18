import unittest
from traits.api import HasTraits, Dict
class MyOtherClass(HasTraits):
    """ A dummy HasTraits class with a Dict """
    d = Dict({'a': 'apple', 'b': 'banana', 'c': 'cherry', 'd': 'durian'})