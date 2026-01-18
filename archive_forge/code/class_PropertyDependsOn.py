import unittest
from traits.api import (
class PropertyDependsOn(HasTraits):
    sum = Property(depends_on='ref.[int1,int2,int3]')
    ref = Instance(ArgCheckBase, ())
    pcalls = Int(0)
    calls = Int(0)
    exp_old = Any
    exp_new = Any
    tc = Any

    @cached_property
    def _get_sum(self):
        self.pcalls += 1
        r = self.ref
        return r.int1 + r.int2 + r.int3

    def _sum_changed(self, old, new):
        self.calls += 1
        self.tc.assertEqual(old, self.exp_old)
        self.tc.assertEqual(new, self.exp_new)