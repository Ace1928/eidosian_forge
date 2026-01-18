import unittest
from traits.api import HasTraits, Str, Int
from traits.testing.unittest_tools import UnittestTools
class TraitsObject(HasTraits):
    string = Str
    integer = Int