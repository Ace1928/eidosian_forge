import unittest
from traits.api import (
class BaseInstance(HasTraits):
    ref = Instance(HasTraits)
    calls = Dict({x: 0 for x in range(5)})
    exp_object = Any
    exp_name = Any
    dst_name = Any
    exp_old = Any
    exp_new = Any
    dst_new = Any
    tc = Any