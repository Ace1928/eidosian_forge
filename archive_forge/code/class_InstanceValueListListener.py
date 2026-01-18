import unittest
from traits.api import (
class InstanceValueListListener(BaseInstance):
    ref = Instance(ArgCheckList, ())

    @on_trait_change('ref.value[]')
    def arg_check0(self):
        self.calls[0] += 1

    @on_trait_change('ref.value[]')
    def arg_check1(self, new):
        self.calls[1] += 1
        self.tc.assertEqual(new, self.dst_new)

    @on_trait_change('ref.value[]')
    def arg_check2(self, name, new):
        self.calls[2] += 1
        self.tc.assertEqual(name, self.dst_name)
        self.tc.assertEqual(new, self.dst_new)

    @on_trait_change('ref.value[]')
    def arg_check3(self, object, name, new):
        self.calls[3] += 1
        self.tc.assertIs(object, self.exp_object)
        self.tc.assertEqual(name, self.exp_name)
        self.tc.assertEqual(new, self.exp_new)

    @on_trait_change('ref.value[]')
    def arg_check4(self, object, name, old, new):
        self.calls[4] += 1
        self.tc.assertIs(object, self.exp_object)
        self.tc.assertEqual(name, self.exp_name)
        self.tc.assertEqual(old, self.exp_old)
        self.tc.assertEqual(new, self.exp_new)