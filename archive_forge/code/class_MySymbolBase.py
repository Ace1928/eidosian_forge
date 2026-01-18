from symengine import symbols, sin, sinh, have_numpy, have_llvm, cos, Symbol
from symengine.test_utilities import raises
import pickle
import unittest
class MySymbolBase(Symbol):

    def __init__(self, name, attr):
        super().__init__(name=name)
        self.attr = attr

    def __eq__(self, other):
        if not isinstance(other, MySymbolBase):
            return False
        return self.name == other.name and self.attr == other.attr