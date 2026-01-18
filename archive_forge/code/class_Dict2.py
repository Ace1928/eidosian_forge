import unittest
from traits.api import (
class Dict2(HasTraits):
    refs = Dict(Int, ArgCheckBase)
    calls = Int(0)
    exp_new = Any
    tc = Any

    @on_trait_change('refs.value')
    def arg_check1(self, new):
        self.calls += 1
        self.tc.assertEqual(new, self.exp_new)