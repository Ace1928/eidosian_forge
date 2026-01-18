import unittest
from llvmlite import ir
from llvmlite import binding as llvm
from llvmlite.tests import TestCase
from . import refprune_proto as proto
class TestFanout(BaseTestByIR):
    """More complex cases are tested in TestRefPrunePass
    """
    refprune_bitmask = llvm.RefPruneSubpasses.FANOUT
    fanout_1 = '\ndefine void @main(i8* %ptr, i1 %cond) {\nbb_A:\n    call void @NRT_incref(i8* %ptr)\n    br i1 %cond, label %bb_B, label %bb_C\nbb_B:\n    call void @NRT_decref(i8* %ptr)\n    ret void\nbb_C:\n    call void @NRT_decref(i8* %ptr)\n    ret void\n}\n'

    def test_fanout_1(self):
        mod, stats = self.check(self.fanout_1)
        self.assertEqual(stats.fanout, 3)
    fanout_2 = '\ndefine void @main(i8* %ptr, i1 %cond, i8** %excinfo) {\nbb_A:\n    call void @NRT_incref(i8* %ptr)\n    br i1 %cond, label %bb_B, label %bb_C\nbb_B:\n    call void @NRT_decref(i8* %ptr)\n    ret void\nbb_C:\n    call void @NRT_decref(i8* %ptr)\n    br label %bb_B                      ; illegal jump to other decref\n}\n'

    def test_fanout_2(self):
        mod, stats = self.check(self.fanout_2)
        self.assertEqual(stats.fanout, 0)
    fanout_3 = '\ndefine void @main(i8* %ptr, i1 %cond) {\nbb_A:\n    call void @NRT_incref(i8* %ptr)\n    call void @NRT_incref(i8* %ptr)\n    br i1 %cond, label %bb_B, label %bb_C\nbb_B:\n    call void @NRT_decref(i8* %ptr)\n    call void @NRT_decref(i8* %ptr)\n    call void @NRT_decref(i8* %ptr)\n    ret void\nbb_C:\n    call void @NRT_decref(i8* %ptr)\n    call void @NRT_decref(i8* %ptr)\n    ret void\n}\n'

    def test_fanout_3(self):
        mod, stats = self.check(self.fanout_3)
        self.assertEqual(stats.fanout, 6)

    def test_fanout_3_limited(self):
        mod, stats = self.check(self.fanout_3, subgraph_limit=1)
        self.assertEqual(stats.fanout, 0)