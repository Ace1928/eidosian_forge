import unittest
from llvmlite import ir
from llvmlite import binding as llvm
from llvmlite.tests import TestCase
from . import refprune_proto as proto
class TestFanoutRaise(BaseTestByIR):
    refprune_bitmask = llvm.RefPruneSubpasses.FANOUT_RAISE
    fanout_raise_1 = '\ndefine i32 @main(i8* %ptr, i1 %cond, i8** %excinfo) {\nbb_A:\n    call void @NRT_incref(i8* %ptr)\n    br i1 %cond, label %bb_B, label %bb_C\nbb_B:\n    call void @NRT_decref(i8* %ptr)\n    ret i32 0\nbb_C:\n    store i8* null, i8** %excinfo, !numba_exception_output !0\n    ret i32 1\n}\n!0 = !{i1 true}\n'

    def test_fanout_raise_1(self):
        mod, stats = self.check(self.fanout_raise_1)
        self.assertEqual(stats.fanout_raise, 2)
    fanout_raise_2 = '\ndefine i32 @main(i8* %ptr, i1 %cond, i8** %excinfo) {\nbb_A:\n    call void @NRT_incref(i8* %ptr)\n    br i1 %cond, label %bb_B, label %bb_C\nbb_B:\n    call void @NRT_decref(i8* %ptr)\n    ret i32 0\nbb_C:\n    store i8* null, i8** %excinfo, !numba_exception_typo !0      ; bad metadata\n    ret i32 1\n}\n\n!0 = !{i1 true}\n'

    def test_fanout_raise_2(self):
        mod, stats = self.check(self.fanout_raise_2)
        self.assertEqual(stats.fanout_raise, 0)
    fanout_raise_3 = '\ndefine i32 @main(i8* %ptr, i1 %cond, i8** %excinfo) {\nbb_A:\n    call void @NRT_incref(i8* %ptr)\n    br i1 %cond, label %bb_B, label %bb_C\nbb_B:\n    call void @NRT_decref(i8* %ptr)\n    ret i32 0\nbb_C:\n    store i8* null, i8** %excinfo, !numba_exception_output !0\n    ret i32 1\n}\n\n!0 = !{i32 1}       ; ok; use i32\n'

    def test_fanout_raise_3(self):
        mod, stats = self.check(self.fanout_raise_3)
        self.assertEqual(stats.fanout_raise, 2)
    fanout_raise_4 = '\ndefine i32 @main(i8* %ptr, i1 %cond, i8** %excinfo) {\nbb_A:\n    call void @NRT_incref(i8* %ptr)\n    br i1 %cond, label %bb_B, label %bb_C\nbb_B:\n    ret i32 1    ; BAD; all tails are raising without decref\nbb_C:\n    ret i32 1    ; BAD; all tails are raising without decref\n}\n\n!0 = !{i1 1}\n'

    def test_fanout_raise_4(self):
        mod, stats = self.check(self.fanout_raise_4)
        self.assertEqual(stats.fanout_raise, 0)
    fanout_raise_5 = '\ndefine i32 @main(i8* %ptr, i1 %cond, i8** %excinfo) {\nbb_A:\n    call void @NRT_incref(i8* %ptr)\n    br i1 %cond, label %bb_B, label %bb_C\nbb_B:\n    call void @NRT_decref(i8* %ptr)\n    br label %common.ret\nbb_C:\n    store i8* null, i8** %excinfo, !numba_exception_output !0\n    br label %common.ret\ncommon.ret:\n    %common.ret.op = phi i32 [ 0, %bb_B ], [ 1, %bb_C ]\n    ret i32 %common.ret.op\n}\n!0 = !{i1 1}\n'

    def test_fanout_raise_5(self):
        mod, stats = self.check(self.fanout_raise_5)
        self.assertEqual(stats.fanout_raise, 2)