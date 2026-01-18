import unittest
from traits.api import (
class ArgCheckBase(HasTraits):
    value = Int(0)
    int1 = Int(0, test=True)
    int2 = Int(0)
    int3 = Int(0, test=True)
    tint1 = Int(0)
    tint2 = Int(0, test=True)
    tint3 = Int(0)
    calls = Int(0)
    tc = Any