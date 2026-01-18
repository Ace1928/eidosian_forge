import unittest
from traits.api import (
class Dict3(HasTraits):
    refs = Dict(Int, ArgCheckBase)
    calls = Int(0)
    exp_name = Any
    exp_new = Any
    tc = Any

    @on_trait_change('refs.value')
    def arg_check2(self, name, new):
        self.calls += 1
        self.tc.assertEqual(name, self.exp_name)
        self.tc.assertEqual(new, self.exp_new)