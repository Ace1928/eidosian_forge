import re
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, skip_on_cudasim, CUDATestCase
from llvmlite import ir
@skip_on_cudasim('This is testing CUDA backend code generation')
class TestConstStringCodegen(unittest.TestCase):

    def test_const_string(self):
        from numba.cuda.descriptor import cuda_target
        from numba.cuda.cudadrv.nvvm import llvm_to_ptx
        targetctx = cuda_target.target_context
        mod = targetctx.create_module('')
        textstring = 'A Little Brown Fox'
        gv0 = targetctx.insert_const_string(mod, textstring)
        targetctx.insert_const_string(mod, textstring)
        res = re.findall('@\\"__conststring__.*internal.*constant.*\\[19\\s+x\\s+i8\\]', str(mod))
        self.assertEqual(len(res), 1)
        fnty = ir.FunctionType(ir.IntType(8).as_pointer(), [])
        fn = ir.Function(mod, fnty, 'test_insert_const_string')
        builder = ir.IRBuilder(fn.append_basic_block())
        res = builder.addrspacecast(gv0, ir.PointerType(ir.IntType(8)), 'generic')
        builder.ret(res)
        matches = re.findall('@\\"__conststring__.*internal.*constant.*\\[19\\s+x\\s+i8\\]', str(mod))
        self.assertEqual(len(matches), 1)
        fn = ir.Function(mod, fnty, 'test_insert_string_const_addrspace')
        builder = ir.IRBuilder(fn.append_basic_block())
        res = targetctx.insert_string_const_addrspace(builder, textstring)
        builder.ret(res)
        matches = re.findall('@\\"__conststring__.*internal.*constant.*\\[19\\s+x\\s+i8\\]', str(mod))
        self.assertEqual(len(matches), 1)
        ptx = llvm_to_ptx(str(mod)).decode('ascii')
        matches = list(re.findall('\\.const.*__conststring__', ptx))
        self.assertEqual(len(matches), 1)