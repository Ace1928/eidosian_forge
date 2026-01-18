import math
import os
import platform
import sys
import re
import numpy as np
from numba import njit
from numba.core import types
from numba.core.runtime import (
from numba.core.extending import intrinsic, include_path
from numba.core.typing import signature
from numba.core.imputils import impl_ret_untracked
from llvmlite import ir
import llvmlite.binding as llvm
from numba.core.unsafe.nrt import NRT_get_api
from numba.tests.support import (EnableNRTStatsMixin, TestCase, temp_directory,
from numba.core.registry import cpu_target
import unittest
class TestRefCtPruning(unittest.TestCase):
    sample_llvm_ir = '\ndefine i32 @"MyFunction"(i8** noalias nocapture %retptr, { i8*, i32 }** noalias nocapture %excinfo, i8* noalias nocapture readnone %env, double %arg.vt.0, double %arg.vt.1, double %arg.vt.2, double %arg.vt.3, double %arg.bounds.0, double %arg.bounds.1, double %arg.bounds.2, double %arg.bounds.3, i8* %arg.xs.0, i8* nocapture readnone %arg.xs.1, i64 %arg.xs.2, i64 %arg.xs.3, double* nocapture readonly %arg.xs.4, i64 %arg.xs.5.0, i64 %arg.xs.6.0, i8* %arg.ys.0, i8* nocapture readnone %arg.ys.1, i64 %arg.ys.2, i64 %arg.ys.3, double* nocapture readonly %arg.ys.4, i64 %arg.ys.5.0, i64 %arg.ys.6.0, i8* %arg.aggs_and_cols.0.0, i8* nocapture readnone %arg.aggs_and_cols.0.1, i64 %arg.aggs_and_cols.0.2, i64 %arg.aggs_and_cols.0.3, i32* nocapture %arg.aggs_and_cols.0.4, i64 %arg.aggs_and_cols.0.5.0, i64 %arg.aggs_and_cols.0.5.1, i64 %arg.aggs_and_cols.0.6.0, i64 %arg.aggs_and_cols.0.6.1) local_unnamed_addr {\nentry:\ntail call void @NRT_incref(i8* %arg.xs.0)\ntail call void @NRT_incref(i8* %arg.ys.0)\ntail call void @NRT_incref(i8* %arg.aggs_and_cols.0.0)\n%.251 = icmp sgt i64 %arg.xs.5.0, 0\nbr i1 %.251, label %B42.preheader, label %B160\n\nB42.preheader:                                    ; preds = %entry\n%0 = add i64 %arg.xs.5.0, 1\nbr label %B42\n\nB42:                                              ; preds = %B40.backedge, %B42.preheader\n%lsr.iv3 = phi i64 [ %lsr.iv.next, %B40.backedge ], [ %0, %B42.preheader ]\n%lsr.iv1 = phi double* [ %scevgep2, %B40.backedge ], [ %arg.xs.4, %B42.preheader ]\n%lsr.iv = phi double* [ %scevgep, %B40.backedge ], [ %arg.ys.4, %B42.preheader ]\n%.381 = load double, double* %lsr.iv1, align 8\n%.420 = load double, double* %lsr.iv, align 8\n%.458 = fcmp ole double %.381, %arg.bounds.1\n%not..432 = fcmp oge double %.381, %arg.bounds.0\n%"$phi82.1.1" = and i1 %.458, %not..432\nbr i1 %"$phi82.1.1", label %B84, label %B40.backedge\n\nB84:                                              ; preds = %B42\n%.513 = fcmp ole double %.420, %arg.bounds.3\n%not..487 = fcmp oge double %.420, %arg.bounds.2\n%"$phi106.1.1" = and i1 %.513, %not..487\nbr i1 %"$phi106.1.1", label %B108.endif.endif.endif, label %B40.backedge\n\nB160:                                             ; preds = %B40.backedge, %entry\ntail call void @NRT_decref(i8* %arg.ys.0)\ntail call void @NRT_decref(i8* %arg.xs.0)\ntail call void @NRT_decref(i8* %arg.aggs_and_cols.0.0)\nstore i8* null, i8** %retptr, align 8\nret i32 0\n\nB108.endif.endif.endif:                           ; preds = %B84\n%.575 = fmul double %.381, %arg.vt.0\n%.583 = fadd double %.575, %arg.vt.1\n%.590 = fptosi double %.583 to i64\n%.630 = fmul double %.420, %arg.vt.2\n%.638 = fadd double %.630, %arg.vt.3\n%.645 = fptosi double %.638 to i64\ntail call void @NRT_incref(i8* %arg.aggs_and_cols.0.0)              ; GONE 1\ntail call void @NRT_decref(i8* null)                                ; GONE 2\ntail call void @NRT_incref(i8* %arg.aggs_and_cols.0.0), !noalias !0 ; GONE 3\n%.62.i.i = icmp slt i64 %.645, 0\n%.63.i.i = select i1 %.62.i.i, i64 %arg.aggs_and_cols.0.5.0, i64 0\n%.64.i.i = add i64 %.63.i.i, %.645\n%.65.i.i = icmp slt i64 %.590, 0\n%.66.i.i = select i1 %.65.i.i, i64 %arg.aggs_and_cols.0.5.1, i64 0\n%.67.i.i = add i64 %.66.i.i, %.590\n%.84.i.i = mul i64 %.64.i.i, %arg.aggs_and_cols.0.5.1\n%.87.i.i = add i64 %.67.i.i, %.84.i.i\n%.88.i.i = getelementptr i32, i32* %arg.aggs_and_cols.0.4, i64 %.87.i.i\n%.89.i.i = load i32, i32* %.88.i.i, align 4, !noalias !3\n%.99.i.i = add i32 %.89.i.i, 1\nstore i32 %.99.i.i, i32* %.88.i.i, align 4, !noalias !3\ntail call void @NRT_decref(i8* %arg.aggs_and_cols.0.0), !noalias !0 ; GONE 4\ntail call void @NRT_decref(i8* %arg.aggs_and_cols.0.0)              ; GONE 5\nbr label %B40.backedge\n\nB40.backedge:                                     ; preds = %B108.endif.endif.endif, %B84, %B42\n%scevgep = getelementptr double, double* %lsr.iv, i64 1\n%scevgep2 = getelementptr double, double* %lsr.iv1, i64 1\n%lsr.iv.next = add i64 %lsr.iv3, -1\n%.294 = icmp sgt i64 %lsr.iv.next, 1\nbr i1 %.294, label %B42, label %B160\n}\n    '

    def test_refct_pruning_op_recognize(self):
        input_ir = self.sample_llvm_ir
        input_lines = list(input_ir.splitlines())
        before_increfs = [ln for ln in input_lines if 'NRT_incref' in ln]
        before_decrefs = [ln for ln in input_lines if 'NRT_decref' in ln]
        output_ir = nrtopt._remove_redundant_nrt_refct(input_ir)
        output_lines = list(output_ir.splitlines())
        after_increfs = [ln for ln in output_lines if 'NRT_incref' in ln]
        after_decrefs = [ln for ln in output_lines if 'NRT_decref' in ln]
        self.assertNotEqual(before_increfs, after_increfs)
        self.assertNotEqual(before_decrefs, after_decrefs)
        pruned_increfs = set(before_increfs) - set(after_increfs)
        pruned_decrefs = set(before_decrefs) - set(after_decrefs)
        combined = pruned_increfs | pruned_decrefs
        self.assertEqual(combined, pruned_increfs ^ pruned_decrefs)
        pruned_lines = '\n'.join(combined)
        for i in [1, 2, 3, 4, 5]:
            gone = '; GONE {}'.format(i)
            self.assertIn(gone, pruned_lines)
        self.assertEqual(len(list(pruned_lines.splitlines())), len(combined))

    @unittest.skip('Pass removed as it was buggy. Re-enable when fixed.')
    def test_refct_pruning_with_branches(self):
        """testcase from #2350"""

        @njit
        def _append_non_na(x, y, agg, field):
            if not np.isnan(field):
                agg[y, x] += 1

        @njit
        def _append(x, y, agg, field):
            if not np.isnan(field):
                if np.isnan(agg[y, x]):
                    agg[y, x] = field
                else:
                    agg[y, x] += field

        @njit
        def append(x, y, agg, field):
            _append_non_na(x, y, agg, field)
            _append(x, y, agg, field)

        @njit(no_cpython_wrapper=True)
        def extend(arr, field):
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    append(j, i, arr, field)
        extend.compile('(f4[:,::1], f4)')
        llvmir = str(extend.inspect_llvm(extend.signatures[0]))
        refops = list(re.finditer('(NRT_incref|NRT_decref)\\([^\\)]+\\)', llvmir))
        self.assertEqual(len(refops), 0)

    @linux_only
    @x86_only
    def test_inline_asm(self):
        """The InlineAsm class from llvmlite.ir has no 'name' attr the refcount
        pruning pass should be tolerant to this"""
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        llvm.initialize_native_asmparser()

        @intrinsic
        def bar(tyctx, x, y):

            def codegen(cgctx, builder, sig, args):
                arg_0, arg_1 = args
                fty = ir.FunctionType(ir.IntType(32), [ir.IntType(32), ir.IntType(32)])
                mul = builder.asm(fty, 'mov $2, $0; imul $1, $0', '=&r,r,r', (arg_0, arg_1), name='asm_mul', side_effect=False)
                return impl_ret_untracked(cgctx, builder, sig.return_type, mul)
            return (signature(types.int32, types.int32, types.int32), codegen)

        @njit(['int32(int32)'])
        def foo(x):
            x += 1
            z = bar(x, 2)
            return z
        self.assertEqual(foo(10), 22)